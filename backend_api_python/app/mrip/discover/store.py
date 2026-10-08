from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, ContextManager, Sequence

from app.mrip.discover.types import Kind, RankedItem


@dataclass(frozen=True, slots=True)
class StoredItem:
    id: int
    as_of: date
    kind: Kind
    subject: str
    headline: str
    magnitude: float
    score: float
    components: dict[str, float]
    details: dict[str, Any]
    data_quality: dict[str, Any]
    modeled: bool
    policy_version: str
    status: str


class DiscoverStore:
    def __init__(self, connect: Callable[[], ContextManager[Any]]) -> None:
        self._connect = connect

    def save(self, items: Sequence[RankedItem], policy_version: str) -> int:
        written = 0
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                for item in items:
                    obs = item.observation
                    cur.execute("INSERT INTO mrip_discover_items (as_of,kind,subject,headline,magnitude,score,components,details,data_quality,modeled,policy_version) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s) ON CONFLICT (as_of,kind,subject) DO UPDATE SET headline=EXCLUDED.headline,magnitude=EXCLUDED.magnitude,score=EXCLUDED.score,components=EXCLUDED.components,details=EXCLUDED.details,data_quality=EXCLUDED.data_quality,modeled=EXCLUDED.modeled,policy_version=EXCLUDED.policy_version", (obs.as_of, obs.kind.value, obs.subject, obs.headline, obs.magnitude, item.score, json.dumps(dict(item.components)), json.dumps(dict(obs.details), default=str), json.dumps(dict(obs.data_quality), default=str), obs.modeled, policy_version))
                    written += 1
            finally:
                cur.close()
            conn.commit()
        return written

    def feed(self, as_of: date | None = None, *, limit: int = 50, kinds: Sequence[Kind] | None = None, include_dismissed: bool = False) -> list[StoredItem]:
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                where = ["as_of = COALESCE(%s, (SELECT MAX(as_of) FROM mrip_discover_items))"]
                params: list[Any] = [as_of]
                if not include_dismissed: where.append("status = 'open'")
                if kinds is not None: where.append("kind = ANY(%s::text[])"); params.append([k.value for k in kinds])
                params.append(limit)
                cur.execute("SELECT id,as_of,kind,subject,headline,magnitude,score,components,details,data_quality,modeled,policy_version,status FROM mrip_discover_items WHERE " + " AND ".join(where) + " ORDER BY score DESC LIMIT %s", params)
                rows = cur.fetchall()
            finally: cur.close()
        return [StoredItem(int(r['id']),r['as_of'],Kind(r['kind']),r['subject'],r['headline'],float(r['magnitude']),float(r['score']),dict(r['components'] or {}),dict(r['details'] or {}),dict(r['data_quality'] or {}),bool(r['modeled']),r['policy_version'],r['status']) for r in rows]

    def dismiss(self, item_id: int) -> bool:
        """Mark an item dismissed. Returns False when no item has that id."""
        with self._connect() as conn:
            cur=conn.cursor()
            try:
                cur.execute("UPDATE mrip_discover_items SET status='dismissed' WHERE id=%s",(item_id,))
                found = cur.rowcount > 0
            finally: cur.close()
            conn.commit()
        return found

    def recent_counts(self, as_of: date, days: int = 7) -> dict[tuple[str,str],int]:
        with self._connect() as conn:
            cur=conn.cursor()
            try:
                cur.execute("SELECT kind,subject,COUNT(DISTINCT as_of) AS days_seen FROM mrip_discover_items WHERE as_of < %s AND as_of >= %s - (%s * INTERVAL '1 day') GROUP BY kind,subject",(as_of,as_of,days))
                rows=cur.fetchall()
            finally: cur.close()
        return {(r['kind'],r['subject']):int(r['days_seen']) for r in rows}
