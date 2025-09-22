"""
Core curiosity engine for question generation, novelty scoring and contradiction detection.

The CuriosityEngine orchestrates language models, retrieval and NLI models to
produce a sequence of questions about a topic.  It keeps track of how novel
each question is relative to an existing corpus and uses a bounded loop to
terminate exploration when new questions stop contributing meaningful novelty.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Dict, List, Tuple

from .models import ModelRouter
from .retrieval import Retriever
from .nli_contradiction import NLIDetector


class CuriosityEngine:
    """Engine for generating and evaluating questions using LLMs and retrieval."""

    def __init__(
        self,
        model_spec: str,
        retriever: Retriever,
        nli_name: str,
        novelty_threshold: float = 0.35,
        max_round_seconds: float = 25.0,
        max_rounds: int = 3,
    ) -> None:
        self.router = ModelRouter(model_spec)
        self.retriever = retriever
        self.nli = NLIDetector(nli_name)
        self.nov_thr = novelty_threshold
        self.max_s = max_round_seconds
        self.max_r = max_rounds

        # Prompt templates used when querying the LLM
        self.rubric = (
            "Ask succinct expert-level questions (why/what if/how). Hunt contradictions and edge cases. "
            "Avoid trivia."
        )
        self.prompt_template = (
            "{rubric}\nContext:\n- {ctx}\n\n"
            "Produce up to {n} distinct questions (<=25 words). Number them."
        )

    def _novelty(self, question: str) -> float:
        """
        Compute a novelty score for a question based on its distance to existing
        knowledge snippets.  A higher value indicates a more novel question.

        The score is calculated as 1 minus the mean of exp(-distance) over the
        top k retrieval results.  Distances are squared Euclidean distances from
        the FAISS index; exp(-d) converts distances to a similarity-like measure
        bounded between 0 and 1.
        """
        hits = self.retriever.search(question, k=5)
        if not hits:
            return 0.5  # Default novelty if no knowledge to compare against
        sims = [math.exp(-max(d, 0.0)) for _, d in hits]
        # Lower similarity => more novel
        novelty = 1.0 - (sum(sims) / len(sims))
        # Clamp to [0, 1]
        return max(0.0, min(1.0, novelty))

    async def generate_questions(self, topic: str, n: int = 6) -> List[Dict[str, Any]]:
        """
        Generate a list of candidate questions for a given topic.

        This method retrieves contextual sentences related to the topic, sends a
        prompt to the LLM asking for distinct expert-level questions, parses
        the numbered list and computes novelty for each question.  The result
        is sorted in descending order of novelty.

        Parameters
        ----------
        topic : str
            Topic for which to generate questions.  This may be a previous
            question if called from the exploration loop.
        n : int, optional
            Number of questions to request from the LLM.  Defaults to 6.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries containing the question text and its
            novelty score, sorted descending by novelty.
        """
        # Retrieve context sentences to seed the LLM.  These provide
        # background information and anchor the model's generation.
        context_hits = self.retriever.search(topic, k=6)
        context_lines = [text for text, _ in context_hits]
        context_block = "\n- ".join(context_lines)
        # Build the full prompt
        prompt = self.prompt_template.format(rubric=self.rubric, ctx=context_block, n=n)
        # Query the model.  We request asynchronous completions from the router.
        raw_output = await self.router.completions(prompt)
        # Parse the numbered list of questions.  Accept lines that begin with
        # a numeral followed by a dot or parenthesis, or lines of reasonable length.
        questions: List[Dict[str, Any]] = []
        for line in raw_output.splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove leading numbering if present
            prefix_removed = False
            if (line[0].isdigit() and "." in line[:3]) or (line[0].isdigit() and line[1] in {')', '.'}):
                parts = line.split(".", 1) if "." in line[:3] else line.split(")", 1)
                if len(parts) == 2:
                    line = parts[1].strip()
                    prefix_removed = True
            if len(line) < 6:
                continue
            novelty = self._novelty(line)
            questions.append({"q": line, "novelty": novelty})
        # Sort by novelty descending
        questions.sort(key=lambda x: x["novelty"], reverse=True)
        return questions

    async def bounded_explore(self, seed: str, per_round: int = 6) -> Dict[str, Any]:
        """
        Perform a bounded exploration starting from a seed question or topic.

        The engine will generate batches of questions, evaluate their novelty
        and continue exploring the most novel question until either the
        specified number of rounds is reached or the allotted time expires.

        Parameters
        ----------
        seed : str
            Initial topic or question to explore.
        per_round : int, optional
            Number of questions to request from the LLM in each round.  Defaults to 6.

        Returns
        -------
        dict
            A dictionary containing the list of accepted questions (the trail)
            and the elapsed time of the session.
        """
        start = time.time()
        trail: List[Dict[str, Any]] = []
        current_seed = seed
        for _ in range(self.max_r):
            if time.time() - start > self.max_s:
                break
            batch = await self.generate_questions(current_seed, n=per_round)
            # Filter questions that exceed the novelty threshold
            fresh = [q for q in batch if q["novelty"] >= self.nov_thr]
            if not fresh:
                break
            # Append to trail and set the most novel question as the next seed
            trail.extend(fresh)
            current_seed = fresh[0]["q"]
        return {
            "trail": trail,
            "elapsed_s": round(time.time() - start, 2),
        }

    def dissonance_scan(self, texts: List[str], threshold: float = 0.65) -> List[Dict[str, Any]]:
        """
        Detect strongly contradictory pairs of sentences within a list.

        Each pair of sentences is evaluated by the NLI model.  If the
        contradiction probability is above `threshold` the pair is recorded
        along with the probability.

        Parameters
        ----------
        texts : List[str]
            List of sentences or questions to compare pairwise.
        threshold : float, optional
            Minimum probability for a pair to be considered contradictory.  Defaults to 0.65.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries with keys 'a', 'b' and 'contradiction'.
        """
        contradictions: List[Dict[str, Any]] = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = texts[i], texts[j]
                score = self.nli.contradiction(a, b)
                if score >= threshold:
                    contradictions.append({"a": a, "b": b, "contradiction": round(score, 3)})
        return contradictions