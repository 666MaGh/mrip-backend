"""EnsembleProvider: weighted average of member quantile forecasts (Vincentization).

Averaging quantile functions keeps every quantile ordered, so no crossing is
introduced. Weights are explicit and versioned; ``inverse_loss_weights`` derives
them from walk-forward evaluation (lower pinball loss -> higher weight) so they
are validated empirically rather than assumed.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from app.mrip.forecast.base import BaseForecastProvider
from app.mrip.forecast.types import QUANTILES, EvaluationReport, ForecastProvider, ForecastRequest, ForecastResult, ForecastUnavailable


def inverse_loss_weights(reports: Mapping[str, EvaluationReport]) -> dict[str, float]:
    """Weights proportional to 1 / mean pinball loss, normalised to sum 1."""
    if not reports:
        raise ValueError("no evaluation reports")
    if any(r.mean_pinball_loss <= 0 for r in reports.values()):
        raise ValueError("pinball loss must be positive")
    raw = {name: 1.0 / r.mean_pinball_loss for name, r in reports.items()}
    total = sum(raw.values())
    return {name: w / total for name, w in raw.items()}


class EnsembleProvider(BaseForecastProvider):
    name = "ensemble"

    def __init__(self, members: Sequence[tuple[ForecastProvider, float]], min_members: int = 1) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        if any(w <= 0 for _, w in members):
            raise ValueError("weights must be positive")
        if not 1 <= min_members <= len(members):
            raise ValueError("min_members must be between 1 and the number of members")
        total = sum(w for _, w in members)
        self._members = [(p, w / total) for p, w in members]
        self._min_members = min_members

    @property
    def weights(self) -> dict[str, float]:
        return {p.version(): w for p, w in self._members}

    def version(self) -> str:
        parts = ",".join(f"{p.version()}:{w:.4f}" for p, w in self._members)
        return f"ensemble-v1[{parts}]"

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        results: list[tuple[ForecastResult, float]] = []
        warnings: list[str] = []
        for provider, weight in self._members:
            try:
                results.append((provider.forecast(request), weight))
            except ForecastUnavailable as exc:
                warnings.append(f"{provider.version()} unavailable: {exc}")
        if len(results) < self._min_members:
            raise ForecastUnavailable(f"only {len(results)} of {len(self._members)} members available: " + "; ".join(warnings))
        total = sum(w for _, w in results)
        combined = {q: sum(r.return_quantiles[q] * w for r, w in results) / total for q in QUANTILES}
        for r, _ in results:
            warnings.extend(f"{r.provider}: {w}" for w in r.warnings)
        return ForecastResult(
            provider=self.name,
            provider_version=self.version(),
            symbol=request.symbol,
            as_of=request.as_of,
            horizon_days=request.horizon_days,
            last_price=request.last_price,
            return_quantiles=combined,
            n_obs=min(r.n_obs for r, _ in results),
            warnings=tuple(warnings),
        )
