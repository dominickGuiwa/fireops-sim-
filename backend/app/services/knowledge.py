"""Retrieval-augmented field manual for the simulator.

A deliberately small, dependency-free RAG stack: the corpus is a synthetic
in-repo field manual about this simulator, the index is TF-IDF with cosine
similarity built in NumPy, and answers are extractive (the best-matching
sentences of the best-matching chunk, always cited). There is no generative
model and no external service; every answer can be traced to its chunk. The
manual describes only this synthetic simulator — it is not real firefighting
or emergency guidance.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np

# Each chunk is one retrievable unit with a stable title used for citations.
FIELD_MANUAL: list[dict[str, str]] = [
    {
        "title": "Mission and scoring",
        "text": (
            "The mission is to help the synthetic fire go out before the step limit while "
            "keeping all three synthetic sites safe. The reward is a penalty budget: new fire "
            "spread, site damage, resource use, and active fire each subtract points, so a "
            "better run loses fewer points. There is no bonus for stalling; success means the "
            "fire is out and every site is preserved."
        ),
    },
    {
        "title": "Abstract actions",
        "text": (
            "Six abstract actions exist. Hold spends nothing. Protect north, south, east, or "
            "west applies temporary protection to that half of the map and costs one resource. "
            "Recon costs one resource and briefly improves the sensor observation. Protection "
            "decays after a few steps, so timing matters more than spamming one sector."
        ),
    },
    {
        "title": "Wind and fire spread",
        "text": (
            "Fire spreads to cardinally adjacent cells with a probability shaped by fuel, "
            "moisture, wind, and protection. Downwind spread is more likely and upwind spread "
            "is less likely. A wind change or wind drift can invalidate a protection plan, so "
            "re-check the wind reading after chaos events."
        ),
    },
    {
        "title": "Sensors and partial observability",
        "text": (
            "Agents never see the true map. Configured reliability controls how many cells "
            "report fire data each step, and visible cells can still carry bounded noise. The "
            "wind reading can be wrong or unknown. Recon temporarily raises effective "
            "reliability, which is often the best move when most of the map is hidden."
        ),
    },
    {
        "title": "Chaos failures",
        "text": (
            "Injected chaos can take sensors offline, degrade sensors or communications, "
            "change the wind, or make response resources unavailable. Failures are seeded and "
            "auditable. Robust behavior means acting sensibly while these failures are active, "
            "not predicting them."
        ),
    },
    {
        "title": "Hard-mode features",
        "text": (
            "Hard-mode configuration can enable ember jumps that ignite spot fires two or "
            "three cells downwind, wind drift that rotates the wind mid-episode, transient "
            "sensor dropout that blanks a whole observation frame, and observation delay that "
            "shows stale fire data. These features exist to make training genuinely difficult "
            "and to punish policies that assume a stationary, fully visible world."
        ),
    },
    {
        "title": "Training difficulty presets",
        "text": (
            "Headless training supports standard, hard, and brutal presets. Standard uses the "
            "current configuration unchanged. Hard enables mild ember jumps, wind drift, and "
            "sensor dropout while sampling sensor reliability between 50 and 100 percent per "
            "episode. Brutal raises those probabilities, adds one step of observation delay, "
            "and samples reliability between 25 and 90 percent."
        ),
    },
    {
        "title": "Learning policy",
        "text": (
            "The learning policy is transparent tabular Q-learning over a coarse summary of "
            "the degraded observation: the dominant visible fire sector, a visibility bucket, "
            "and a resource bucket. It updates its value table from step rewards, explores "
            "with a decaying epsilon, and persists only for the server session. Manual moves "
            "also teach it off-policy."
        ),
    },
    {
        "title": "Agentic failure analysis",
        "text": (
            "Every executed step is compared against the true world to flag agentic failures: "
            "perception gaps where most burning cells were invisible, misdirected protection "
            "of the wrong sector, holding while fire burned, acting with no resources left, "
            "and committing to a protect action on a wrong wind reading. The truth grid is "
            "used only for this analysis and is never given to a policy."
        ),
    },
    {
        "title": "PPO and DQN baselines",
        "text": (
            "Optional deep reinforcement learning baselines train against the same Gymnasium "
            "environment. PPO uses Stable-Baselines3; the DQN script implements a replay "
            "buffer, target network, and epsilon-greedy exploration in PyTorch. Both save "
            "metadata with seeds, hyperparameters, and artifact hashes so results are "
            "reproducible. A short training run is a functional demo, not evidence of general "
            "robustness."
        ),
    },
    {
        "title": "Reading the failure panel",
        "text": (
            "When the failure panel flags many perception gaps, lower reliability is hiding "
            "the fire front; consider Recon before committing resources. Repeated misdirected "
            "protection under chaos usually means the policy is trusting stale or noisy "
            "readings. Idle-while-burning flags on the learning policy early in training are "
            "expected and should fade as its value table improves."
        ),
    },
    {
        "title": "Scope and safety",
        "text": (
            "The simulator is a simplified synthetic research environment. Its maps, "
            "facilities, failures, and dynamics are fictional abstractions. Nothing in this "
            "manual is operational guidance for real emergencies, evacuation, firefighting, "
            "or any life-safety decision."
        ),
    },
]

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can for from in is it its of on or so that the this to "
    "with while when where which".split()
)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOPWORDS
    ]


class KnowledgeIndex:
    """TF-IDF index with cosine scoring over the field-manual chunks."""

    def __init__(self, corpus: list[dict[str, str]]) -> None:
        self.corpus = corpus
        documents = [tokenize(chunk["title"] + " " + chunk["text"]) for chunk in corpus]
        self.vocabulary: dict[str, int] = {}
        for tokens in documents:
            for token in tokens:
                self.vocabulary.setdefault(token, len(self.vocabulary))
        counts = np.zeros((len(documents), len(self.vocabulary)), dtype=np.float64)
        for row, tokens in enumerate(documents):
            for token in tokens:
                counts[row, self.vocabulary[token]] += 1.0
        document_frequency = np.count_nonzero(counts, axis=0)
        # Smoothed IDF keeps every seen term finite and positive.
        self.idf = np.log((1.0 + len(documents)) / (1.0 + document_frequency)) + 1.0
        weighted = counts * self.idf
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.matrix = weighted / norms

    def _query_vector(self, query: str) -> np.ndarray | None:
        tokens = [token for token in tokenize(query) if token in self.vocabulary]
        if not tokens:
            return None
        vector = np.zeros(len(self.vocabulary), dtype=np.float64)
        for token in tokens:
            vector[self.vocabulary[token]] += 1.0
        vector *= self.idf
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else None

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        vector = self._query_vector(query)
        if vector is None:
            return []
        scores = self.matrix @ vector
        order = np.argsort(scores)[::-1][:k]
        results = []
        for rank, index in enumerate(order, start=1):
            score = float(scores[int(index)])
            if score <= 0.0:
                continue
            chunk = self.corpus[int(index)]
            results.append(
                {
                    "rank": rank,
                    "title": chunk["title"],
                    "text": chunk["text"],
                    "score": round(score, 4),
                }
            )
        return results

    def extractive_answer(self, query: str, top_chunk: dict[str, Any]) -> str:
        """Return the best-overlapping sentences of the top chunk, verbatim."""

        query_tokens = set(tokenize(query))
        sentences = re.split(r"(?<=[.!?])\s+", top_chunk["text"])
        scored = sorted(
            (
                (len(query_tokens & set(tokenize(sentence))), position, sentence)
                for position, sentence in enumerate(sentences)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        chosen = [item for item in scored[:2] if item[0] > 0]
        if not chosen:
            return top_chunk["text"]
        return " ".join(sentence for _, _, sentence in sorted(chosen, key=lambda item: item[1]))


@lru_cache(maxsize=1)
def get_index() -> KnowledgeIndex:
    return KnowledgeIndex(FIELD_MANUAL)


def answer_query(query: str, k: int = 3) -> dict[str, Any]:
    index = get_index()
    results = index.search(query, k=k)
    answer = index.extractive_answer(query, results[0]) if results else None
    return {
        "query": query,
        "results": results,
        "answer": answer,
        "note": (
            "Extractive retrieval over the synthetic simulator field manual (TF-IDF + "
            "cosine). No generative model; every answer is quoted from a cited chunk. "
            "Simulator guidance only — never real emergency advice."
        ),
    }
