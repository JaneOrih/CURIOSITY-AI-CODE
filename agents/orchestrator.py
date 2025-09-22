"""
High-level orchestrator that coordinates the curiosity engine.

The Orchestrator encapsulates the logic for a single exploration session.
It runs the bounded exploration to generate a sequence of novel questions and
then performs a dissonance scan on the resulting questions to surface
potential contradictions.  The combined result is returned to the API
layer for presentation to the user.
"""
from __future__ import annotations

from typing import Any, Dict, List

from engine.curiosity_engine import CuriosityEngine


class Orchestrator:
    """Coordinate exploration and contradiction detection."""

    def __init__(self, engine: CuriosityEngine) -> None:
        self.engine = engine

    async def run_cycle(self, topic: str) -> Dict[str, Any]:
        """
        Execute a full curiosity cycle for the given topic.

        The cycle involves:

        1. Running the bounded exploration starting from the topic to gather
           a trail of novel questions.
        2. Selecting a subset of these questions (up to eight) to scan for
           contradictions using the NLI model.

        Parameters
        ----------
        topic : str
            Seed topic from which to start the exploration.

        Returns
        -------
        dict
            A dictionary containing the original topic, session details and
            a list of detected contradictions.
        """
        session = await self.engine.bounded_explore(seed=topic, per_round=6)
        # Extract the question texts for dissonance scanning.  Limit to 8
        # questions to keep the number of pairwise comparisons manageable.
        questions = [item["q"] for item in session["trail"]][:8]
        dissonance = self.engine.dissonance_scan(questions)
        return {
            "topic": topic,
            "session": session,
            "dissonance": dissonance,
        }