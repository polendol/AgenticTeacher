import argparse, random, re
import numpy as np, pandas as pd, torch
import math
from datasets import load_dataset, Dataset
from huggingface_hub import list_datasets
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from peft import LoraConfig, get_peft_model, TaskType

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
TEXT_COLS = ["text"]
LABEL_COLS = ["cefr_level", "level", "cefr", "label"]
CHUNK_WORDS = 250

MODELS = {
    "deberta": dict(
        name="microsoft/deberta-v3-base",
        targets=["query_proj", "key_proj", "value_proj", "dense"],
        save=["classifier", "pooler"]), #TEST: {'test_loss': 0.46324828267097473, 'test_qwk': 0.7706145974737674, 'test_adjacent_acc': 0.9694947569113441, 'test_macro_f1': 0.5760069348195234, 'test_exact_acc': 0.559580552907531, 'test_runtime': 82.6138, 'test_samples_per_second': 12.698, 'test_steps_per_second': 0.399}
    "modernbert": dict(
        name="answerdotai/ModernBERT-base",
        targets=["Wqkv", "Wo", "Wi"],
        save=["head", "classifier"]),
}

# ---------------------------------------------------------------- data
def pick(cols, options):
    for o in options:
        if o in cols:
            return o
    raise ValueError(f"None of {options} found. Columns are: {cols}")

def set_level(x):
    """
    Make sure that any variation in level, such as B1/B2, or even b1+ are uppercased and matched to the labels in LEVELS
     dictionary as the first two symbols they contain, as long as they have two characters, one being A,B or C and
     another being 1 or 2
    """
    m = re.match(r"\s*([ABC][12])", str(x).upper())
    return LEVELS.index(m.group(1)) if m else None

def load_universalcefr(repos):
    if not repos:
        repos = [d.id for d in list_datasets(author="UniversalCEFR") if d.id.endswith("_en")] #take all UniversalCEFR datasets that finish in 'en' to make sure all texts in English are included
    print("English datasets used:", repos)
    frames = []
    for r in repos:
        ds = load_dataset(r)
        df = pd.concat([ds[s].to_pandas() for s in ds])
        tcol, lcol = pick(df.columns, TEXT_COLS), pick(df.columns, LABEL_COLS)
        out = pd.DataFrame({"text": df[tcol], "y": df[lcol].map(set_level), "src": r}) #creates a dataframe with the texts, its level and its source
        frames.append(out.dropna()) #append the out dataset but only when there is no None as level
    df = pd.concat(frames).drop_duplicates("text").reset_index(drop=True)
    df["y"] = df["y"].astype(int)
    print(df["y"].map(lambda i: LEVELS[i]).value_counts().sort_index(), "\n")
    return df

def chunk(df):
    rows = []
    for text, y in zip(df.text, df.y):
        w = text.split()
        for i in range(0, max(len(w), 1), CHUNK_WORDS):
            piece = " ".join(w[i:i + CHUNK_WORDS])
            if len(piece.split()) >= 15:               # skip tiny leftovers
                rows.append({"text": piece, "y": y})
    return pd.DataFrame(rows)

def degrade(text):
    """Make some training texts look like YouTube auto-captions."""
    return re.sub(r"[^\w\s']", "", text.lower())

def balance(df):
    """Resample every level to the median class size."""
    n = int(df.y.value_counts().median())
    parts = [g.sample(n, replace=len(g) < n, random_state=42)
             for _, g in df.groupby("y")]
    return pd.concat(parts).reset_index(drop=True)

# ---------------------------------------------------------------- metrics
def compute_metrics(p):
    pred = np.clip(np.rint(p.predictions.squeeze()), 0, 5).astype(int)
    y = p.label_ids.astype(int)
    return {
        "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
        "adjacent_acc": float(np.mean(np.abs(pred - y) <= 1)),
        "macro_f1": f1_score(y, pred, average="macro"),
        "exact_acc": float(np.mean(pred == y)),
    }

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--repos", nargs="*", default=None, help="override dataset repos")
    ap.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max_len", type=int, default=384)
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    cfg = MODELS[args.model]

    # split by DOCUMENT first, then chunk, so chunks of one text never leak across splits
    docs = load_universalcefr(args.repos)
    strat = docs.y if docs.y.value_counts().min() >= 3 else None
    train_d, rest = train_test_split(docs, test_size=0.2, stratify=strat, random_state=42)
    strat = rest.y if rest.y.value_counts().min() >= 2 else None
    val_d, test_d = train_test_split(rest, test_size=0.5, stratify=strat, random_state=42)

    train, val, test = chunk(train_d), chunk(val_d), chunk(test_d)
    train = balance(train)
    mask = np.random.rand(len(train)) < 0.3
    train.loc[mask, "text"] = train.loc[mask, "text"].map(degrade)
    print(f"chunks: train={len(train)} val={len(val)} test={len(test)}")

    tok = AutoTokenizer.from_pretrained(cfg["name"])
    def prep(df):
        ds = Dataset.from_pandas(df.rename(columns={"y": "labels"}), preserve_index=False)
        ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                    batched=True, remove_columns=["text"])
        return ds.map(lambda b: {"labels": [float(x) for x in b["labels"]]}, batched=True)
    train_ds, val_ds, test_ds = prep(train), prep(val), prep(test)

    # regression: one output neuron, level = round(score)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["name"], num_labels=1, problem_type="regression")
    if not args.full:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=32, lora_alpha=64, lora_dropout=0.1,
            target_modules=cfg["targets"], modules_to_save=cfg["save"]))
        model.print_trainable_parameters()

    lr = args.lr or (2e-2 if False else (2e-4 if not args.full else 2e-5))
    out = f"models/{args.model}-trained-pefted-cefr"
    steps_per_epoch = math.ceil(len(train_ds) / args.batch)
    warmup = int(0.1 * steps_per_epoch * args.epochs)
    targs = TrainingArguments(
        output_dir=f"{out}-checkpoints", num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=32,
        learning_rate=lr, warmup_steps=warmup, weight_decay=0.01,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        eval_strategy="epoch", save_strategy="epoch",   # older transformers: evaluation_strategy
        load_best_model_at_end=True, metric_for_best_model="qwk",
        save_total_limit=1, logging_steps=50, report_to="none")
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    trainer.train()

    # final test on held-out documents
    res = trainer.predict(test_ds)
    print("\nTEST:", res.metrics)
    pred = np.clip(np.rint(res.predictions.squeeze()), 0, 5).astype(int)
    print(pd.DataFrame(confusion_matrix(res.label_ids.astype(int), pred, labels=range(6)),
                       index=LEVELS, columns=LEVELS))

    # merge LoRA into the base model so inference needs no peft
    if not args.full:
        model = model.merge_and_unload()
    model.save_pretrained(out); tok.save_pretrained(out)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()