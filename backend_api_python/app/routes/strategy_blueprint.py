"""Shared strategy blueprint.

Strategy routes are split across modules while preserving the same blueprint
and URL prefixes registered by `app.openapi.register`.

MRIP (ADR-0001): unless ``TRADING_ENABLED=true`` only the research/authoring
rules below are mounted. Every other rule (live strategy CRUD, start/stop,
positions, trades, ledger, grid orders, broker account views, executors,
notifications, ...) is dropped at decoration time, so it never reaches the
URL map or the OpenAPI spec. The allow-list fails closed: new upstream
strategy routes stay hidden until explicitly listed here.
"""
from app.openapi.blueprint import HumanBlueprint
from app.utils.trading_gate import trading_enabled

# Rules (prefix match) that stay available without trading enabled.
RESEARCH_RULE_PREFIXES: tuple[str, ...] = (
    "/strategies/generate",
    "/strategies/verify",
    "/strategies/ai-workspace",
    "/strategies/script-",
    "/strategy-assets",
)


class StrategyBlueprint(HumanBlueprint):
    """HumanBlueprint that omits execution rules when trading is disabled."""

    def route(self, rule: str, *, methods=None, **options):
        if not trading_enabled() and not rule.startswith(RESEARCH_RULE_PREFIXES):
            return lambda f: f
        return super().route(rule, methods=methods, **options)


strategy_blp = StrategyBlueprint('strategy', __name__)
