"""
Natural Language Inference (NLI) based contradiction detection.

This module wraps a Hugging Face transformers model that has been fine
tuned for natural language inference.  It computes the probability that
two sentences contradict each other by evaluating them with the model.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import os
import torch
os.environ.setdefault("TRANSFORMERS_NO_TF_IMPORTS", "1")
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NLIDetector:
    """Detect contradictions between pairs of sentences using an NLI model."""

    def __init__(self, model_name: str = "roberta-large-mnli") -> None:
        """
        Load the specified NLI model and tokenizer.  The default model is
        ``roberta-large-mnli`` which performs well on general language.  If
        working in specialised domains (e.g. biomedical) you may wish to
        substitute a domain-specific model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def contradiction(self, sentence_a: str, sentence_b: str) -> float:
        """
        Compute the probability that `sentence_a` contradicts `sentence_b`.

        The returned value is the probability mass associated with the
        'contradiction' class of the NLI model's output distribution.
        """
        inputs = self.tokenizer(sentence_a, sentence_b, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        # The classes are ordered as [contradiction, neutral, entailment]
        return float(probs[0].item())