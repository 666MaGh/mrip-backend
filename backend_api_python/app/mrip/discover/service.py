from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Mapping, Sequence

import pandas as pd

from app.mrip.discover.detectors import DiscoverPolicy, cot_events, gamma_events, peer_events, regime_events, relationship_events
from app.mrip.discover.ranking import RankingPolicy, rank
from app.mrip.discover.types import Observation
from app.mrip.options.analysis import analyze_options
from app.mrip.prices.select import select_pair
from app.mrip.stats.service import prices_to_series
from app.mrip.stats.transforms import log_returns


@dataclass(frozen=True, slots=True)
class DiscoverRunSummary:
    items_saved: int
    per_detector: dict[str, int]
    errors: dict[str, str]
    as_of: date


class DiscoverService:
    def __init__(self, *, graph: Any, price_store: Any, snapshot_store: Any, discover_store: Any, evidence_store: Any = None, regime_engine: Any = None, cot_engine: Any = None, policy: DiscoverPolicy = DiscoverPolicy(), rank_policy: RankingPolicy = RankingPolicy()) -> None:
        self.graph, self.price_store, self.snapshot_store, self.discover_store = graph, price_store, snapshot_store, discover_store
        self.evidence_store, self.regime_engine, self.cot_engine = evidence_store, regime_engine, cot_engine
        self.policy, self.rank_policy = policy, rank_policy

    def run(self, as_of: date, *, option_symbols: Sequence[str] | None = None) -> DiscoverRunSummary:
        observations: list[Observation] = []; errors: dict[str,str] = {}
        def group(name: str, fn: Any) -> None:
            try: observations.extend(fn())
            except Exception as exc: errors[name] = f"{type(exc).__name__}: {exc}"
        def options() -> list[Observation]:
            symbols = option_symbols if option_symbols is not None else getattr(self.snapshot_store, "symbols", lambda: [])()
            end = datetime.combine(as_of, time.max, timezone.utc); out=[]
            for symbol in symbols:
                current=self.snapshot_store.latest_before(symbol,end)
                if current is None: continue
                previous=self.snapshot_store.latest_before(symbol,current.snapshot_timestamp-timedelta(seconds=1))
                out.extend(gamma_events(analyze_options(current),analyze_options(previous) if previous else None,as_of,self.policy))
            return out
        group("options",options)
        if self.regime_engine is not None:
            group("regime",lambda: regime_events(self.regime_engine.at(as_of),self.regime_engine.at(as_of-timedelta(days=7)),as_of,self.policy))
        if self.cot_engine is not None:
            def cot() -> list[Observation]:
                result=[]
                for market in self.cot_engine.markets(): result.extend(cot_events(self.cot_engine.snapshot(market,as_of),as_of,self.policy))
                return result
            group("cot",cot)
        nodes: dict[int,Any]={}
        def relationships() -> list[Observation]:
            out=[]
            for edge in self.graph.list_active_edges(limit=10000):
                src=nodes.get(edge.src_id) or self.graph.get_node_by_id(edge.src_id); dst=nodes.get(edge.dst_id) or self.graph.get_node_by_id(edge.dst_id)
                if not src or not dst: continue
                nodes[src.id]=src; nodes[dst.id]=dst
                def symbol(node: Any) -> str | None:
                    val=node.attributes.get("series",{}); return val.get("symbol") if isinstance(val,dict) else None
                a,b=symbol(src),symbol(dst)
                if not a or not b: continue
                need=self.policy.relationship_fit_days+self.policy.relationship_recent_days
                pair=select_pair(self.price_store,a,b,as_of,min_overlap=need+1)
                if pair is None: continue  # no common provider with enough overlap: leg unavailable, nothing mixed
                obs=relationship_events(f"{a}->{b}",log_returns(prices_to_series(pair.left,as_of)),log_returns(prices_to_series(pair.right,as_of)),as_of,self.policy,edge.status.value)
                if self.evidence_store:
                    from app.mrip.evidence.types import RelationshipRef
                    ref=RelationshipRef(src.key and __import__('app.mrip.relationships.types',fromlist=['NodeKey']).NodeKey(src.node_type,src.key),__import__('app.mrip.relationships.types',fromlist=['NodeKey']).NodeKey(dst.node_type,dst.key),edge.relation_type)
                    summary=self.evidence_store.summary(ref)
                    counts={"support":summary.by_stance.get(__import__('app.mrip.evidence.types',fromlist=['Stance']).Stance.SUPPORT,0),"contradict":summary.by_stance.get(__import__('app.mrip.evidence.types',fromlist=['Stance']).Stance.CONTRADICT,0)}
                    obs=[replace(o,evidence=counts) for o in obs]
                out.extend(obs)
            return out
        if hasattr(self.graph,"list_active_edges"): group("relationships",relationships)
        liquidity: dict[str, float] = {}
        def peers() -> list[Observation]:
            all_obs=[]
            for universe in ("SP500","OMXSLC"):
                ns=self.graph.list_nodes_in_universe(universe); returns={}; sectors={}; dollar_volumes: dict[str,float] = {}
                for node in ns:
                    info=node.attributes.get("series",{}); symbol=info.get("symbol") if isinstance(info,dict) else None
                    if not symbol: continue
                    series=self.price_store.series("cboe" if universe=="SP500" else "yahoo",symbol,end=as_of)
                    if series:
                        returns[symbol]=log_returns(prices_to_series(series,as_of)); sectors[symbol]=str(node.attributes.get("sector","UNKNOWN"))
                        observations = [(bar.close * bar.volume) for bar in series.bars[-20:] if bar.close is not None and bar.volume is not None]
                        if observations:
                            dollar_volumes[symbol] = sum(observations) / len(observations)
                ordered = sorted(dollar_volumes.values())
                if ordered:
                    for symbol, value in dollar_volumes.items():
                        liquidity[symbol] = sum(candidate <= value for candidate in ordered) / len(ordered)
                all_obs.extend(peer_events(returns,sectors,as_of,self.policy))
            return all_obs
        if hasattr(self.graph,"list_nodes_in_universe"): group("peers",peers)
        if liquidity:
            observations = [replace(obs, liquidity=liquidity[obs.subject]) if obs.subject in liquidity else obs for obs in observations]
        recent=self.discover_store.recent_counts(as_of)
        ranked=rank(observations,recent=recent,policy=self.rank_policy)
        counts:dict[str,int]={}
        for item in ranked: counts[item.observation.kind.value]=counts.get(item.observation.kind.value,0)+1
        saved=self.discover_store.save(ranked,self.rank_policy.version)
        return DiscoverRunSummary(saved,counts,errors,as_of)
