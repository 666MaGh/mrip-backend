"""Point-in-time options snapshot storage: pure round trip and PostgreSQL behaviour (opt-in)."""
import dataclasses
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path as FsPath

import pytest

from app.mrip.data.models import Latency, OptionContract, OptionsChainSnapshot, Provenance
from app.mrip.options.analysis import analyze_options
from app.mrip.options.snapshots import OptionsSnapshotStore, SnapshotError, decode_contracts, encode_contracts

DB = pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")
MIGRATION = FsPath(__file__).resolve().parents[2] / "migrations" / "mrip_20261001_options_snapshots.sql"
T1 = datetime(2026, 10, 1, 19, 0, tzinfo=timezone.utc)
PROV = Provenance("cboe", "openbb", "cboe.options.chains", T1, Latency.DELAYED, "5.0.0")


def chain(ts=T1, oi_scale=1, price=100.0, oi_date=None):
    contracts = tuple(
        OptionContract(
            contract_symbol=f"X{k}{kind}", expiry=date(2026, 10, 16), strike=float(k), option_type=kind,
            open_interest=1000 * oi_scale, volume=10, implied_volatility=0.2 + k / 1000, delta=0.5, gamma=0.03,
            theta=-0.1, vega=0.2, rho=0.05,
        )
        for k in range(90, 111) for kind in ("call", "put")
    )
    return OptionsChainSnapshot(
        "TEST", price, datetime(2026, 10, 1, 14, 58), ts, oi_date, contracts,
        dataclasses.replace(PROV, fetched_at=ts),
    )


def test_contract_encoding_round_trips_exactly_and_deterministically():
    contracts = chain().contracts
    blob = encode_contracts(contracts)
    assert decode_contracts(blob) == contracts
    assert encode_contracts(contracts) == blob  # same bytes -> stable hash
    with_gaps = (OptionContract(None, date(2026, 10, 16), 100.0, "call"),)
    assert decode_contracts(encode_contracts(with_gaps)) == with_gaps


def test_unsupported_format_is_rejected():
    import gzip

    with pytest.raises(SnapshotError, match="unsupported"):
        decode_contracts(gzip.compress(b'{"v": 99, "contracts": []}'))


class _NoDb:
    def __call__(self):
        raise AssertionError("must not touch the database")


def test_save_validates_before_touching_the_database():
    store = OptionsSnapshotStore(_NoDb())
    with pytest.raises(SnapshotError, match="timezone-aware"):
        store.save(dataclasses.replace(chain(), snapshot_timestamp=datetime(2026, 10, 1, 19, 0)))
    with pytest.raises(SnapshotError, match="empty"):
        store.save(dataclasses.replace(chain(), contracts=()))
    with pytest.raises(SnapshotError, match="timezone-aware"):
        store.latest_before("TEST", datetime(2026, 10, 1))


@pytest.fixture()
def store():
    from app.utils import db

    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(MIGRATION.read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_options_snapshots RESTART IDENTITY")
        conn.commit()
        cur.close()
    return OptionsSnapshotStore(db.get_db_connection)


@DB
def test_round_trip_through_the_database_preserves_the_snapshot_and_its_analysis(store):
    original = chain(oi_date=date(2026, 9, 30))
    loaded = store.load(store.save(original))
    assert loaded == original
    assert analyze_options(loaded) == analyze_options(original)


@DB
def test_saving_the_same_snapshot_twice_is_idempotent(store):
    a, b = store.save(chain()), store.save(chain())
    assert a == b and store.timestamps("TEST") == [T1]


@DB
def test_latest_before_never_returns_a_later_snapshot(store):
    early, late = chain(T1, oi_scale=1), chain(T1 + timedelta(days=1), oi_scale=5)
    store.save(early), store.save(late)
    assert store.latest_before("TEST", T1 - timedelta(seconds=1)) is None
    assert store.latest_before("TEST", T1).contracts[0].open_interest == 1000
    assert store.latest_before("TEST", T1 + timedelta(hours=23)).contracts[0].open_interest == 1000  # not the later one
    assert store.latest_before("TEST", T1 + timedelta(days=1)).contracts[0].open_interest == 5000
    assert store.latest_before("OTHER", T1 + timedelta(days=9)) is None


@DB
def test_corrupted_payload_is_detected(store):
    from app.utils import db

    snap_id = store.save(chain())
    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE mrip_options_snapshots SET payload_sha256 = repeat('0', 64) WHERE id = %s", (snap_id,))
        conn.commit()
        cur.close()
    with pytest.raises(SnapshotError, match="integrity"):
        store.load(snap_id)
    with pytest.raises(SnapshotError, match="no snapshot"):
        store.load(snap_id + 1000)
