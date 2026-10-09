# Third-party notices

This repository contains code only. It does not include model weights,
datasets, or YouTube content.

## Models
- CEFR classifiers (DeBERTa-v3 and ModernBERT LoRA fine-tunes), hosted on
  Hugging Face under CC-BY-NC-SA-4.0:
  https://huggingface.co/Polender/DeBERTa-CEFR-classifier
- Base models: microsoft/deberta-v3-base (MIT) and ModernBERT (Apache-2.0).
- Exercise generation: Qwen3 4B, run locally through Ollama (Apache-2.0).

## Training data (for the classifiers)
UniversalCEFR English subsets (Imperial et al., EMNLP 2025), each under its own license:
cambridge_exams_en (CC-BY-NC-SA-4.0), elg_cefr_en (CC-BY-NC-4.0),
cefr_sp_en (CC-BY-NC-SA-4.0), icle500_en (CC-BY-NC-4.0),
cefr_asag_en (CC-BY-NC-SA-4.0), readme_en (CC-BY-NC-SA-4.0).



## Other tools
- spaCy model en_core_web_sm is downloaded separately under its own license.
- Ollama is a separate program with its own license.
