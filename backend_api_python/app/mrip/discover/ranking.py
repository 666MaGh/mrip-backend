from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from app.mrip.discover.types import Observation, RankedItem


@dataclass(frozen=True, slots=True)
class RankingPolicy:
    version: str = "discover-rank-v0-uncalibrated"
    magnitude: float = 0.30
    evidence: float = 0.15
    novelty: float = 0.15
    reliability: float = 0.10
    confidence: float = 0.10
    liquidity: float = 0.10
    data_quality: float = 0.10

    def __post_init__(self) -> None:
        if abs(sum((self.magnitude, self.evidence, self.novelty, self.reliability, self.confidence, self.liquidity, self.data_quality)) - 1) > 1e-9:
            raise ValueError("ranking weights must sum to 1")


def rank(observations: Sequence[Observation], *, recent: Mapping[tuple[str, str], int] | None = None, policy: RankingPolicy = RankingPolicy()) -> list[RankedItem]:
    weights = {"magnitude": policy.magnitude, "evidence": policy.evidence, "novelty": policy.novelty, "reliability": policy.reliability, "confidence": policy.confidence, "liquidity": policy.liquidity, "data_quality": policy.data_quality}
    results: list[RankedItem] = []
    for obs in observations:
        unknown: list[str] = []
        values: dict[str, float] = {"magnitude": obs.magnitude}
        if obs.evidence is None:
            unknown.append("evidence"); values["evidence"] = .5
        else:
            support, contradict = obs.evidence.get("support", 0), obs.evidence.get("contradict", 0)
            value = support / (support + contradict) if support + contradict else .5
            if support and contradict:
                value -= .25
            values["evidence"] = min(1.0, max(0.0, value))
        days = (recent or {}).get((obs.kind.value, obs.subject))
        values["novelty"] = 1.0 if days is None else max(.2, 1 - days / 7)
        for name in ("reliability", "confidence", "liquidity"):
            value = getattr(obs, name)
            if value is None:
                unknown.append(name); values[name] = .5
            else:
                values[name] = min(1.0, max(0.0, value))
        values["data_quality"] = min(1.0, max(0.0, float(obs.data_quality.get("completeness", 1.0))))
        score = round(100 * sum(weights[k] * values[k] for k in weights), 4)
        results.append(RankedItem(obs, score, values, tuple(unknown)))
    return sorted(results, key=lambda item: (-item.score, item.observation.kind.value, item.observation.subject))
