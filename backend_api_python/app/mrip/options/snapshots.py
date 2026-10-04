"""Point-in-time storage of observed options chains (work 007).

Rows are immutable. ``latest_before`` returns the newest snapshot taken at or
before a given instant, so analyses and backtests can never see a later chain.
There is deliberately no way to build a past chain from current open interest.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from datetime import date, datetime
from typing import Any, Callable, ContextManager

from app.mrip.data.models import Latency, OptionContract, OptionsChainSnapshot, Provenance

_FORMAT = 1


class SnapshotError(Exception):
    """Invalid snapshot operation or corrupted stored payload."""


def encode_contracts(contracts: tuple[OptionContract, ...]) -> bytes:
    rows = [
        [
            c.contract_symbol, c.expiry.isoformat(), c.strike, c.option_type, c.open_interest, c.volume,
            c.implied_volatility, c.delta, c.gamma, c.theta, c.vega, c.rho, c.contract_multiplier,
        ]
        for c in contracts
    ]
    raw = json.dumps({"v": _FORMAT, "contracts": rows}, separators=(",", ":")).encode("utf-8")
    return gzip.compress(raw, mtime=0)  # mtime=0 keeps the bytes (and hash) deterministic


def decode_contracts(payload: bytes) -> tuple[OptionContract, ...]:
    data = json.loads(gzip.decompress(payload))
    if data.get("v") != _FORMAT:
        raise SnapshotError(f"unsupported snapshot format {data.get('v')!r}")
    return tuple(
        OptionContract(
            contract_symbol=r[0], expiry=date.fromisoformat(r[1]), strike=r[2], option_type=r[3],
            open_interest=r[4], volume=r[5], implied_volatility=r[6], delta=r[7], gamma=r[8],
            theta=r[9], vega=r[10], rho=r[11], contract_multiplier=r[12],
        )
        for r in data["contracts"]
    )


_COLUMNS = (
    "id, underlying, snapshot_timestamp, underlying_price, underlying_timestamp, oi_effective_date, "
    "provider, gateway, endpoint, fetched_at, latency, gateway_version, payload, payload_sha256"
)


def _snapshot(row: Any) -> OptionsChainSnapshot:
    payload = bytes(row["payload"])
    if hashlib.sha256(payload).hexdigest() != row["payload_sha256"]:
        raise SnapshotError(f"snapshot {row['id']} failed its integrity check")
    return OptionsChainSnapshot(
        underlying=row["underlying"],
        underlying_price=row["underlying_price"],
        underlying_timestamp=row["underlying_timestamp"],
        snapshot_timestamp=row["snapshot_timestamp"],
        oi_effective_date=row["oi_effective_date"],
        contracts=decode_contracts(payload),
        provenance=Provenance(
            provider=row["provider"], gateway=row["gateway"], endpoint=row["endpoint"],
            fetched_at=row["fetched_at"], latency=Latency(row["latency"]), gateway_version=row["gateway_version"],
        ),
    )


class OptionsSnapshotStore:
    def __init__(self, connect: Callable[[], ContextManager[Any]]) -> None:
        self._connect = connect

    def save(self, snapshot: OptionsChainSnapshot) -> int:
        """Store a snapshot; saving the same (underlying, timestamp, provider) again returns the existing id."""
        if snapshot.snapshot_timestamp.tzinfo is None:
            raise SnapshotError("snapshot_timestamp must be timezone-aware")
        if not snapshot.contracts:
            raise SnapshotError("refusing to store an empty chain")
        payload = encode_contracts(snapshot.contracts)
        p = snapshot.provenance
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO mrip_options_snapshots "
                    "(underlying, snapshot_timestamp, underlying_price, underlying_timestamp, oi_effective_date, "
                    " n_contracts, provider, gateway, endpoint, fetched_at, latency, gateway_version, payload, payload_sha256) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (underlying, snapshot_timestamp, provider) DO NOTHING RETURNING id",
                    (
                        snapshot.underlying, snapshot.snapshot_timestamp, snapshot.underlying_price,
                        snapshot.underlying_timestamp, snapshot.oi_effective_date, len(snapshot.contracts),
                        p.provider, p.gateway, p.endpoint, p.fetched_at, p.latency.value, p.gateway_version,
                        payload, hashlib.sha256(payload).hexdigest(),
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        "SELECT id FROM mrip_options_snapshots "
                        "WHERE underlying = %s AND snapshot_timestamp = %s AND provider = %s",
                        (snapshot.underlying, snapshot.snapshot_timestamp, p.provider),
                    )
                    row = cur.fetchone()
            finally:
                cur.close()
            conn.commit()
        return int(row["id"])

    def load(self, snapshot_id: int) -> OptionsChainSnapshot:
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT " + _COLUMNS + " FROM mrip_options_snapshots WHERE id = %s", (snapshot_id,))
                row = cur.fetchone()
            finally:
                cur.close()
        if row is None:
            raise SnapshotError(f"no snapshot with id {snapshot_id}")
        return _snapshot(row)

    def latest_before(self, underlying: str, as_of: datetime) -> OptionsChainSnapshot | None:
        """Newest snapshot with snapshot_timestamp <= as_of (never a later one)."""
        if as_of.tzinfo is None:
            raise SnapshotError("as_of must be timezone-aware")
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT " + _COLUMNS + " FROM mrip_options_snapshots "
                    "WHERE underlying = %s AND snapshot_timestamp <= %s "
                    "ORDER BY snapshot_timestamp DESC, id DESC LIMIT 1",
                    (underlying, as_of),
                )
                row = cur.fetchone()
            finally:
                cur.close()
        return _snapshot(row) if row else None

    def timestamps(self, underlying: str) -> list[datetime]:
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT snapshot_timestamp FROM mrip_options_snapshots WHERE underlying = %s "
                    "ORDER BY snapshot_timestamp",
                    (underlying,),
                )
                rows = cur.fetchall()
            finally:
                cur.close()
        return [r["snapshot_timestamp"] for r in rows]

    def symbols(self) -> list[str]:
        """Return underlyings with at least one stored options snapshot."""
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT DISTINCT underlying FROM mrip_options_snapshots ORDER BY underlying")
                rows = cur.fetchall()
            finally:
                cur.close()
        return [str(row["underlying"]) for row in rows]
