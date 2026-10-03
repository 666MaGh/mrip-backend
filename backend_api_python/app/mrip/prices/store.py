"""Local price bar storage with incremental sync tracking (work 013)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, ContextManager, Sequence

from app.mrip.data.models import Latency, PriceBar, PriceSeries, Provenance


def collapse_bars_by_date(bars: Sequence[PriceBar]) -> list[PriceBar]:
    """Collapse bars that map to the same bar_date, keeping the last occurrence.

    Invalid bars (close None or <= 0) are removed first. The result is sorted by
    date and ready for insertion without ON CONFLICT collisions.

    Args:
        bars: Sequence of PriceBar objects, potentially with duplicate dates.

    Returns:
        List of PriceBar objects with one per distinct bar_date, sorted by date.
    """
    # Filter out invalid bars first.
    valid_bars = [
        b for b in bars
        if b.close is not None and b.close > 0
    ]

    if not valid_bars:
        return []

    # Collapse by bar_date, keeping the last occurrence (provider's correction).
    bar_by_date: dict[date, PriceBar] = {}
    for bar in valid_bars:
        bar_date = bar.ts.date() if isinstance(bar.ts, datetime) else bar.ts
        bar_by_date[bar_date] = bar

    # Sort by date.
    sorted_bars = sorted(bar_by_date.values(), key=lambda b: b.ts.date() if isinstance(b.ts, datetime) else b.ts)

    return sorted_bars


@dataclass(frozen=True, slots=True)
class SyncState:
    """Current sync status for one (provider, symbol) pair."""

    provider: str
    symbol: str
    last_bar_date: date | None
    last_success_at: datetime | None
    last_attempt_at: datetime
    last_error: str | None
    consecutive_failures: int


class PriceStore:
    """PostgreSQL-backed store of daily price bars and sync state."""

    def __init__(self, connect: Callable[[], ContextManager[Any]]) -> None:
        self._connect = connect

    def upsert_bars(
        self, provider: str, symbol: str, bars: Sequence[PriceBar]
    ) -> int:
        """Store price bars with ON CONFLICT DO UPDATE (provider revisions overwrite).

        Bars whose close is None or <= 0 are skipped and not counted.
        Bars that map to the same bar_date are collapsed, keeping the last occurrence.
        Returns the count of distinct bars actually inserted/updated.
        """
        collapsed_bars = collapse_bars_by_date(bars)
        if not collapsed_bars:
            return 0

        count = 0
        # Write in chunks of 500 rows per transaction.
        chunk_size = 500
        for chunk_start in range(0, len(collapsed_bars), chunk_size):
            chunk = collapsed_bars[chunk_start : chunk_start + chunk_size]
            count += self._upsert_chunk(provider, symbol, chunk)

        return count

    def _upsert_chunk(
        self, provider: str, symbol: str, bars: Sequence[PriceBar]
    ) -> int:
        """Insert/update a single chunk of bars in one transaction."""
        if not bars:
            return 0

        with self._connect() as conn:
            cur = conn.cursor()
            try:
                # Build multi-row VALUES clause.
                placeholders = ", ".join(
                    [
                        (
                            "(%s, %s, %s, %s, %s, %s, %s, %s)"
                            if i == 0
                            else "(%s, %s, %s, %s, %s, %s, %s, %s)"
                        )
                        for i in range(len(bars))
                    ]
                )
                values: list[Any] = []
                for bar in bars:
                    bar_date = bar.ts.date() if isinstance(bar.ts, datetime) else bar.ts
                    values.extend([
                        provider,
                        symbol,
                        bar_date,
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                    ])

                cur.execute(
                    f"""
                    INSERT INTO mrip_price_bars
                    (provider, symbol, bar_date, open, high, low, close, volume)
                    VALUES {placeholders}
                    ON CONFLICT (provider, symbol, bar_date)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        ingested_at = NOW()
                    """,
                    values,
                )
                conn.commit()
                return len(bars)
            finally:
                cur.close()

    def series(
        self,
        provider: str,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> PriceSeries | None:
        """Retrieve price series from the store with optional date range.

        Returns None if no bars exist. Bars are ordered by date ascending.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                # Build WHERE clause for date range.
                where_parts = ["provider = %s", "symbol = %s"]
                params: list[Any] = [provider, symbol]

                if start is not None:
                    where_parts.append("bar_date >= %s")
                    params.append(start)
                if end is not None:
                    where_parts.append("bar_date <= %s")
                    params.append(end)

                where_clause = " AND ".join(where_parts)

                cur.execute(
                    f"""
                    SELECT bar_date, open, high, low, close, volume, ingested_at
                    FROM mrip_price_bars
                    WHERE {where_clause}
                    ORDER BY bar_date ASC
                    """,
                    params,
                )

                rows = cur.fetchall()
            finally:
                cur.close()

        if not rows:
            return None

        # Convert rows to PriceBar objects.
        bars = []
        fetched_at = None
        for row in rows:
            bar_date: date = row["bar_date"]
            ingested_at: datetime = row["ingested_at"]

            # Track the maximum ingested_at for provenance.
            if fetched_at is None or ingested_at > fetched_at:
                fetched_at = ingested_at

            bars.append(
                PriceBar(
                    ts=bar_date,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )
            )

        if fetched_at is None:
            fetched_at = datetime.now(timezone.utc)

        provenance = Provenance(
            provider=provider,
            gateway="mrip-store",
            endpoint="mrip_price_bars",
            fetched_at=fetched_at,
            latency=Latency.UNKNOWN,
        )

        return PriceSeries(
            symbol=symbol,
            interval="1d",
            bars=tuple(bars),
            provenance=provenance,
        )

    def last_bar_date(self, provider: str, symbol: str) -> date | None:
        """Return the most recent bar date, or None if no bars exist."""
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT bar_date FROM mrip_price_bars "
                    "WHERE provider = %s AND symbol = %s "
                    "ORDER BY bar_date DESC LIMIT 1",
                    (provider, symbol),
                )
                row = cur.fetchone()
            finally:
                cur.close()

        return row["bar_date"] if row else None

    def record_attempt(
        self,
        provider: str,
        symbol: str,
        *,
        success: bool,
        last_bar_date: date | None = None,
        error: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Update sync state after an ingestion attempt.

        On success: set last_success_at, update last_bar_date (keep old if new is None),
                    reset consecutive_failures, clear error.
        On failure: increment consecutive_failures, store error (truncated to 500 chars).
        Always: set last_attempt_at.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        with self._connect() as conn:
            cur = conn.cursor()
            try:
                if success:
                    # Upsert on success: reset counters, update bar date if provided.
                    cur.execute(
                        """
                        INSERT INTO mrip_price_sync
                        (provider, symbol, last_bar_date, last_success_at, last_attempt_at,
                         last_error, consecutive_failures)
                        VALUES (%s, %s, %s, %s, %s, NULL, 0)
                        ON CONFLICT (provider, symbol)
                        DO UPDATE SET
                            last_bar_date = COALESCE(%s, mrip_price_sync.last_bar_date),
                            last_success_at = %s,
                            last_attempt_at = %s,
                            last_error = NULL,
                            consecutive_failures = 0
                        """,
                        (provider, symbol, last_bar_date, now, now,
                         last_bar_date, now, now),
                    )
                else:
                    # Upsert on failure: increment failures, store error.
                    error_text = (error[:500] if error else None)
                    cur.execute(
                        """
                        INSERT INTO mrip_price_sync
                        (provider, symbol, last_bar_date, last_success_at, last_attempt_at,
                         last_error, consecutive_failures)
                        VALUES (%s, %s, NULL, NULL, %s, %s, 1)
                        ON CONFLICT (provider, symbol)
                        DO UPDATE SET
                            last_attempt_at = %s,
                            last_error = %s,
                            consecutive_failures = mrip_price_sync.consecutive_failures + 1
                        """,
                        (provider, symbol, now, error_text, now, error_text),
                    )

                conn.commit()
            finally:
                cur.close()

    def sync_states(
        self,
        provider: str,
        symbols: Sequence[str] | None = None,
    ) -> list[SyncState]:
        """Retrieve sync states for one provider, optionally filtered to a symbol list.

        If symbols is None, return all symbols for the provider.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                if symbols is None:
                    cur.execute(
                        """
                        SELECT provider, symbol, last_bar_date, last_success_at,
                               last_attempt_at, last_error, consecutive_failures
                        FROM mrip_price_sync
                        WHERE provider = %s
                        ORDER BY symbol
                        """,
                        (provider,),
                    )
                else:
                    # Filter to specific symbols.
                    cur.execute(
                        """
                        SELECT provider, symbol, last_bar_date, last_success_at,
                               last_attempt_at, last_error, consecutive_failures
                        FROM mrip_price_sync
                        WHERE provider = %s AND symbol = ANY(%s)
                        ORDER BY symbol
                        """,
                        (provider, list(symbols)),
                    )

                rows = cur.fetchall()
            finally:
                cur.close()

        return [
            SyncState(
                provider=row["provider"],
                symbol=row["symbol"],
                last_bar_date=row["last_bar_date"],
                last_success_at=row["last_success_at"],
                last_attempt_at=row["last_attempt_at"],
                last_error=row["last_error"],
                consecutive_failures=row["consecutive_failures"],
            )
            for row in rows
        ]
