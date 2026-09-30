## CEFR classifier training

Trains a small transformer to estimate the CEFR level (A1 to C2) of an English text. The trained model is later used by the agent to analyse YouTube transcripts.

Task: ordinal regression (A1=0 ... C2=5, one output, rounded to a level)
Method: LoRA (PEFT) fine-tuning, merged into the base model after training
Data: English datasets from the UniversalCEFR collection
Models: any Hugging Face encoder you name, e.g. microsoft/deberta-v3-base, answerdotai/ModernBERT-base

## How it works
1. Load data. Downloads the English UniversalCEFR datasets and maps each label (B1, b1+, B1/B2...) to a number 0 to 5. Unusable labels are dropped.
2. Split by document (80% train / 10% val / 10% test, fixed seed). The split happens before chunking, so pieces of the same text never end up in both train and test.
3. Chunk every text into pieces of up to 250 words.
4. Balance the levels. Every level is resampled to the median class size, so rare levels (A1, C2) are not ignored.
5. Augment. 30% of training chunks lose their punctuation and casing, to mimic YouTube auto-captions. Validation and test data are left untouched.
6. Train with LoRA (r=32, all linear layers, regression head fully trained). The best epoch by validation QWK is kept.
7. Evaluate on the held-out test documents and save the metrics.
8. Merge and save. The LoRA weights are merged into the base model, so the saved folder is a standalone model.

If a model has already been trained, training is skipped and the saved one is reused.

## Practical notes
1. Time: a few hours on a mid-range GPU for 3 epochs, depending on dataset size. The first run is the slow one.
2. Out of memory: lower the batch size (16 to 8) or max_len (384 to 256). LoRA already uses less memory than full fine-tuning.
3. Precision: bf16 is used only when the GPU supports it (RTX 30xx and newer, A100). Older cards such as the T4 train in fp32, which is slower. DeBERTa-v3 can be unstable in fp16, so watch for a nan loss if you try it.
4. Dataset scarcity: if A1 or C2 have very few texts, the resampling repeats them. Read the per-level counts printed at the start of training.