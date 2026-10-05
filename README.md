## Teaching Assistant 

This program was made to help teachers plan their classes with audiovisual input. It uses a ttyd interface terminal to interact with teachers. 
It works in two stages:

Stage 1: lexical analysis
Teachers introduces the url of the YT video they want to use in class. The program returns a dictionary of words per lexical category (verb, noun, etc.), as well as the CEFR level of the video. The CEFR classifier uses a fine-tuned (PEFT) DeBERTa model (see CEFRclassifier training and Model)

Stage 2: Agent
In this stage, the teacher interacts with an agent to create exercises suitable for the video and the CEFR level. The teacher can choose among 4 types of exercises and can also indicate the length and the focus of the exercise. 

The agent shuts the program only when the teacher explicitly says so. 

The details about the model, the training, and the agentic flow are listed below. 

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

## Model
Two candidate encoders were fine-tuned and evaluated on the same held-out test set. DeBERTa was selected as the production model.

Candidates compared:
1. DeBERTa encoder, fine-tuned
2. ModernBERT encoder, fine-tuned

Raw outputs:
DeBERTa:
`{'test_loss': 0.46324828267097473,
 'test_qwk': 0.7706145974737674,
 'test_adjacent_acc': 0.9694947569113441,
 'test_macro_f1': 0.5760069348195234,
 'test_exact_acc': 0.559580552907531,
 'test_runtime': 82.6138,
 'test_samples_per_second': 12.698,
 'test_steps_per_second': 0.399}`

ModernBERT:
`{'test_loss': 0.4720533788204193,
 'test_qwk': 0.7576356831703336,
 'test_adjacent_acc': 0.9714013346043852,
 'test_macro_f1': 0.58880494444519,
 'test_exact_acc': 0.567206863679695,
 'test_runtime': 123.8522,
 'test_samples_per_second': 8.47,
 'test_steps_per_second': 0.266}`

Selected model: DeBERTa (fine-tuned for CEFR level prediction)
Reason:
1. Higher QWK (0.771 vs 0.758). CEFR levels are ordinal, so the cost of an error grows with its distance from the true level. QWK penalizes predictions quadratically by that distance, which makes it the most appropriate primary metric for this task. DeBERTa leads by +0.013.
2. Lower test loss (0.463 vs 0.472), indicating slightly better-calibrated predictions overall.
3. Faster inference in our setup (~1.5x). DeBERTa processed 12.7 samples/s versus 8.5 for ModernBERT on the test set, reducing serving cost and latency.
4. Comparable on the remaining metrics. ModernBERT is ahead on exact accuracy (+0.8 pts), macro F1 (+1.3 pts) and adjacent accuracy (+0.2 pts), but these margins are small and do not outweigh the QWK advantage for an ordinal task.

## Agentic flow
The agent runs entirely locally with qwen3:4b through Ollama and is orchestrated with LangGraph. It talks to the teacher, asks what kind of exercises they want, and generates them at the transcript's CEFR level.

The agent is a fixed workflow with a human-in-the-loop pause, not a free-roaming ReAct agent. Small local models are much more reliable when code controls the flow, so most steps are plain Python and the LLM is used only where language generation or interpretation is needed.
It creates 4 types of exercises: multiple choice, gap filling, true or false, and short answer. 
After every set of exercises, the agent asks whether the teacher wants any others. It only ends when the teacher says so. The agent never ends on its own.

Limitations:
1. Context limits. A 4B model with an 8k context sees only the first 12,000 characters of the transcript and the top words per part of speech. Very long videos are covered only partially.
2. Factual quality isn't checked. Validation checks structure and consistency, not whether a multiple-choice answer is truly correct or a true/false item is accurate. A teacher should review the output.
3. Level matching is prompt-based. The model is instructed to write at the CEFR level, but nothing verifies the exercises themselves match it.
4. Exercise types are fixed to the four above. Adding one means updating ExerciseType, the generation prompt and validate.


