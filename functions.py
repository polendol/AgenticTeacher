from youtube_transcript_api import YouTubeTranscriptApi
import re
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy
from collections import defaultdict

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
MODEL_PATH = "models/deberta-trained-pefted-cefr"  # folder saved by the training script
MAX_LEN = 384  # same as training
MIN_WORDS = 15  # same "tiny leftover" threshold as training
BATCH_SIZE = 16

_cefr_model = {}  # cache so the model is loaded only once

nlp = spacy.load("en_core_web_sm") #for the lexical dictionary

def fetch_video_id(url: str) -> str:
    pattern = r"(?:v=|\/|be\/|embed\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return url  #the original url is returned if no matches were found

def fetch_transcript(video_id):
    fetched = YouTubeTranscriptApi().fetch(video_id)
    return " ".join(snippet.text for snippet in fetched)

def clean_transcript(transcript: str) -> str:
    return "".join(char if (char.isalnum() or char.isspace() or char in "!?") else " " for char in transcript)

def words_dictionary(text: str) -> dict[str, list[str]]:
    groups = defaultdict(set)
    for token in nlp(text):
        if token.is_alpha and not token.is_stop:
            groups[token.pos_].add(token.lemma_.lower())
    return {pos: sorted(words) for pos, words in groups.items()}

def _load_cefr_model(): #load the model only once, not everytime, and it is internal to this module
    if not _cefr_model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _cefr_model["tok"] = AutoTokenizer.from_pretrained(MODEL_PATH)
        _cefr_model["model"] = (AutoModelForSequenceClassification
                                .from_pretrained(MODEL_PATH).to(device).eval())
        _cefr_model["device"] = device
    return _cefr_model["tok"], _cefr_model["model"], _cefr_model["device"]


def classify_cefr(transcript: str, chunk_words: int = 250) -> dict:
    """Score a transcript on the CEFR scale with the regression DeBERTa model.

    The transcript is split into `chunk_words`-word chunks (250 = training value),
    each chunk gets a score (0 = A1 ... 5 = C2), and the scores are averaged
    weighted by chunk length, then rounded to a level.
    """
    words = transcript.split()
    if not words:
        raise ValueError("Transcript is empty.")

    chunks = [words[i:i + chunk_words] for i in range(0, len(words), chunk_words)]
    kept = [c for c in chunks if len(c) >= MIN_WORDS]  # drop tiny leftovers, as in training
    chunks = kept or chunks  # but never end up with nothing
    texts = [" ".join(c) for c in chunks]

    tok, model, device = _load_cefr_model()
    scores = []
    with torch.inference_mode():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = tok(texts[i:i + BATCH_SIZE], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(device)
            out = model(**batch).logits.squeeze(-1)
            scores.extend(out.float().cpu().numpy().reshape(-1).tolist())

    scores = np.array(scores)
    weights = np.array([len(c) for c in chunks], dtype=float)
    mean = float(np.average(scores, weights=weights))
    to_level = lambda s: CEFR_LEVELS[int(np.clip(np.rint(s), 0, 5))]

    return {
        "level": to_level(mean),  # e.g. "B2"
        "score": round(mean, 3),  # continuous, 0 (A1) to 5 (C2)
        "chunk_scores": [round(float(s), 3) for s in scores],
        "chunk_levels": [to_level(s) for s in scores],
        "std": round(float(scores.std()), 3),  # high = uneven difficulty
        "n_chunks": len(chunks),
        "n_words": len(words),
    }