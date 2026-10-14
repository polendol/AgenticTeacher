# Third-party notices

This repository contains **code only**. It does not include model weights,
datasets, or YouTube content. Everything listed here is downloaded separately
by each user and remains under its own license.

Licenses were checked on 14 October 2026 against the official sources
(Hugging Face model and dataset pages, the Ollama library page and the
metadata of the spaCy package). Check them again before relying on them.

## Models

| Model | Role | License | Source |
|---|---|---|---|
| Polender/DeBERTa-CEFR-classifier | CEFR classifier used by the application. A LoRA fine-tune of `microsoft/deberta-v3-base`, merged into the base model | **CC-BY-NC-SA-4.0** (non-commercial) | https://huggingface.co/Polender/DeBERTa-CEFR-classifier |
| microsoft/deberta-v3-base | Base model of the classifier | MIT | https://huggingface.co/microsoft/deberta-v3-base |
| answerdotai/ModernBERT-base | Second candidate compared during training. Its fine-tuned model is **not** published | Apache-2.0 | https://huggingface.co/answerdotai/ModernBERT-base |
| Qwen3 4B (`qwen3:4b`) | Exercise generation, run locally through Ollama | Apache-2.0 | https://ollama.com/library/qwen3 |
| spaCy `en_core_web_sm` 3.8.0 | Tokenization, POS tagging and lemmatization. Installed through `requirements.txt` | MIT | https://spacy.io/models/en#en_core_web_sm |

The classifier is released under CC-BY-NC-SA-4.0 because of the licenses of
its training data (below). The code in this repository is MIT, but the project
as a whole is intended for non-commercial use.

The `en_core_web_sm` package metadata lists these training sources: OntoNotes 5
(licensed by Explosion), ClearNLP Constituent-to-Dependency Conversion
(citation only, no code packaged) and WordNet 3.0 (WordNet 3.0 License).

## Training data (for the classifier)

English subsets of UniversalCEFR (Imperial et al., EMNLP 2025,
https://arxiv.org/abs/2506.01419), each under its own license:

| Dataset | License |
|---|---|
| UniversalCEFR/cambridge_exams_en | CC-BY-NC-SA-4.0 |
| UniversalCEFR/elg_cefr_en | CC-BY-NC-4.0 |
| UniversalCEFR/cefr_sp_en | CC-BY-NC-SA-4.0 |
| UniversalCEFR/icle500_en | CC-BY-NC-4.0 |
| UniversalCEFR/cefr_asag_en | CC-BY-NC-SA-4.0 |
| UniversalCEFR/readme_en | CC-BY-NC-SA-4.0 |

The datasets are not included in this repository. The training script downloads
every `_en` dataset of the UniversalCEFR organization at run time. If more
datasets are added there, check their licenses before training or publishing a
model. Please cite UniversalCEFR and the original dataset papers (listed on
each dataset page) if you use these data.

## Software

| Component | Role | License |
|---|---|---|
| youtube-transcript-api | Fetches public YouTube transcripts (uses an undocumented YouTube interface, see below) | MIT |
| Ollama | Local LLM runtime (separate program) | MIT |
| spaCy | NLP library | MIT |
| LangGraph | Agent workflow | MIT |
| langchain-ollama | Connector to Ollama | MIT |
| Pydantic | Data schemas | MIT |
| Hugging Face Transformers, PEFT, Datasets, safetensors, huggingface_hub | Model loading, fine-tuning and data | Apache-2.0 |
| SentencePiece | Tokenizer | Apache-2.0 |
| PyTorch, NumPy, scikit-learn, pandas | Numerics, metrics and data handling | BSD-style |

PyTorch wheels for Linux install NVIDIA CUDA runtime packages, which are
distributed under NVIDIA's own license terms.

## YouTube content

Video transcripts are obtained with an unofficial library that uses an
undocumented part of YouTube, not the official YouTube API. Transcripts belong
to their creators. This project does not store or redistribute them, and
exercises generated from a transcript should not be republished. Users are
responsible for following YouTube's Terms of Service.

## Trademarks

YouTube is a trademark of Google LLC. This is an independent project, not
affiliated with or endorsed by YouTube, Google, Microsoft, Alibaba (Qwen),
Hugging Face, the UniversalCEFR authors, Explosion, LangChain or Ollama. All
product names and trademarks belong to their respective owners.
