"""Throttled, resumable price ingestion with failure tracking (work 013)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Sequence

from app.mrip.data.gateway import DataUnavailable, FinancialDataGateway
from app.mrip.prices.store import PriceStore


@dataclass(frozen=True, slots=True)
class SyncReport:
    """Summary of a price sync run."""

    attempted: int
    succeeded: int
    failed: dict[str, str]
    skipped_fresh: int
    not_attempted: int
    bars_written: int
    halted_reason: str | None


class PriceIngestor:
    """Incremental price ingestion with pacing, circuit breaker, and budget awareness."""

    def __init__(
        self,
        gateway: FinancialDataGateway,
        store: PriceStore,
        *,
        provider: str = "cboe",
        history_start: date = date(2016, 1, 1),
        overlap_days: int = 7,
        min_interval_seconds: float = 2.0,
        max_consecutive_failures: int = 8,
        skip_if_fresh_hours: float = 12.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._gateway = gateway
        self._store = store
        self._provider = provider
        self._history_start = history_start
        self._overlap_days = overlap_days
        self._min_interval_seconds = min_interval_seconds
        self._max_consecutive_failures = max_consecutive_failures
        self._skip_if_fresh_hours = skip_if_fresh_hours
        self._clock = clock
        self._sleep = sleep
        self._now = now

    def sync(
        self,
        symbols: Sequence[str],
        *,
        budget_seconds: float | None = None,
    ) -> SyncReport:
        """Sync price history for a list of symbols with time budget and pacing.

        Returns a SyncReport with ingestion statistics.
        """
        start_time = self._clock()
        now = self._now()

        # Look up sync states for all requested symbols.
        states_by_symbol = {
            state.symbol: state
            for state in self._store.sync_states(self._provider, symbols)
        }

        # Order symbols: never synced first, then by oldest last_success_at.
        ordered_symbols = self._order_symbols(symbols, states_by_symbol, now)

        report = SyncReport(
            attempted=0,
            succeeded=0,
            failed={},
            skipped_fresh=0,
            not_attempted=0,
            bars_written=0,
            halted_reason=None,
        )

        consecutive_failures = 0
        last_call_time: float | None = None

        for symbol in ordered_symbols:
            # Check budget.
            if budget_seconds is not None:
                elapsed = self._clock() - start_time
                if elapsed >= budget_seconds:
                    # Count remaining symbols as not_attempted.
                    report = self._replace_report(
                        report,
                        not_attempted=report.not_attempted + (len(ordered_symbols) - ordered_symbols.index(symbol)),
                        halted_reason="budget exhausted",
                    )
                    break

            # Check freshness.
            state = states_by_symbol.get(symbol)
            if state and state.last_success_at is not None:
                since_success = (now - state.last_success_at).total_seconds() / 3600.0
                if since_success < self._skip_if_fresh_hours:
                    report = self._replace_report(
                        report,
                        skipped_fresh=report.skipped_fresh + 1,
                    )
                    continue

            # Ensure minimum interval between calls (skip pacing before first call).
            if last_call_time is not None:
                time_since_last_call = self._clock() - last_call_time
                if time_since_last_call < self._min_interval_seconds:
                    sleep_duration = self._min_interval_seconds - time_since_last_call
                    self._sleep(sleep_duration)

            # Re-check budget after pacing (in case sleep pushed us over).
            if budget_seconds is not None:
                elapsed = self._clock() - start_time
                if elapsed >= budget_seconds:
                    # Count remaining symbols as not_attempted.
                    remaining_idx = ordered_symbols.index(symbol)
                    report = self._replace_report(
                        report,
                        not_attempted=report.not_attempted + (len(ordered_symbols) - remaining_idx),
                        halted_reason="budget exhausted",
                    )
                    break

            last_call_time = self._clock()

            # Determine start date for this symbol.
            if state and state.last_bar_date is not None:
                start_date = state.last_bar_date - timedelta(days=self._overlap_days)
            else:
                start_date = self._history_start

            # Attempt to fetch and ingest.
            report = self._replace_report(report, attempted=report.attempted + 1)

            try:
                series = self._gateway.price_history(symbol, start=start_date)
                bars = series.bars

                if bars:
                    bars_count = self._store.upsert_bars(self._provider, symbol, bars)
                    max_bar_date = max(
                        (b.ts.date() if isinstance(b.ts, datetime) else b.ts) for b in bars
                    )
                    self._store.record_attempt(
                        self._provider,
                        symbol,
                        success=True,
                        last_bar_date=max_bar_date,
                        now=now,
                    )
                    report = self._replace_report(
                        report,
                        succeeded=report.succeeded + 1,
                        bars_written=report.bars_written + bars_count,
                    )
                    consecutive_failures = 0
                else:
                    # No bars returned; treat as success.
                    self._store.record_attempt(
                        self._provider,
                        symbol,
                        success=True,
                        now=now,
                    )
                    report = self._replace_report(
                        report,
                        succeeded=report.succeeded + 1,
                    )
                    consecutive_failures = 0

            except DataUnavailable as exc:
                error_str = str(exc)
                self._store.record_attempt(
                    self._provider,
                    symbol,
                    success=False,
                    error=error_str,
                    now=now,
                )
                report = self._replace_report(
                    report,
                    failed={**report.failed, symbol: error_str},
                )
                consecutive_failures += 1

                # Check circuit breaker.
                if consecutive_failures >= self._max_consecutive_failures:
                    # Count remaining symbols as not_attempted.
                    remaining_idx = ordered_symbols.index(symbol) + 1
                    report = self._replace_report(
                        report,
                        not_attempted=report.not_attempted + (len(ordered_symbols) - remaining_idx),
                        halted_reason=f"provider appears throttled: {consecutive_failures} consecutive failures",
                    )
                    break

        return report

    def _order_symbols(
        self,
        symbols: Sequence[str],
        states_by_symbol: dict[str, Any],
        now: datetime,
    ) -> list[str]:
        """Order symbols: never-synced first, then by oldest last_success_at."""
        never_synced = []
        synced = []

        for symbol in symbols:
            state = states_by_symbol.get(symbol)
            if state is None or state.last_success_at is None:
                never_synced.append(symbol)
            else:
                synced.append((symbol, state.last_success_at))

        # Sort synced by oldest last_success_at.
        synced.sort(key=lambda x: x[1])
        return never_synced + [s[0] for s in synced]

    @staticmethod
    def _replace_report(
        report: SyncReport,
        **changes: Any,
    ) -> SyncReport:
        """Return a new SyncReport with updated fields."""
        return SyncReport(
            attempted=changes.get("attempted", report.attempted),
            succeeded=changes.get("succeeded", report.succeeded),
            failed=changes.get("failed", report.failed),
            skipped_fresh=changes.get("skipped_fresh", report.skipped_fresh),
            not_attempted=changes.get("not_attempted", report.not_attempted),
            bars_written=changes.get("bars_written", report.bars_written),
            halted_reason=changes.get("halted_reason", report.halted_reason),
        )
