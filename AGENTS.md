# QuantDinger Repository Instructions

This is the canonical instruction file for coding agents working in this repository. Keep tool-specific entrypoints thin and do not duplicate these rules elsewhere.

## Repository map

- `backend_api_python/`: Flask API, services, trading runtime, migrations, and backend tests.
- `mcp_server/`: the tenant-scoped MCP client/server package backed by QuantDinger APIs.
- `docs/`: product, architecture, deployment, API, Agent Gateway, and trading documentation.
- `docker-compose*.yml` and `scripts/`: deployment topology and lifecycle tooling.
- Frontend source is maintained outside this repository. The prebuilt frontend image is configured by the `frontend` service in `docker-compose.yml`.

## Read by task

Read only the documentation relevant to the change:

- Agent Gateway or MCP: [Agent documentation](docs/agent/README.md), [Agent OpenAPI](docs/agent/agent-openapi.json), and [MCP package documentation](mcp_server/README.md).
- Strategy API, backtests, or trading behavior: [Strategy development guide](docs/trading/STRATEGY_DEV_GUIDE.md) and [live-trading safety](docs/trading/LIVE_TRADING_SAFETY.md).
- Runtime ownership, concurrency, Kafka, workers, or durable state: [architecture index](docs/architecture/README.md) and the task-specific document it links.
- HTTP contracts: [API conventions](docs/architecture/API_CONVENTIONS.md) and the applicable OpenAPI document.
- Installation or operations: the root [README](README.md) and the applicable guide under `docs/deployment/`.

## Architecture and contract invariants

- Preserve the process ownership documented in the architecture index. Do not move persistent trading loops into the HTTP backend or let evaluator processes submit exchange orders directly.
- PostgreSQL is the durable source of truth. Kafka transports versioned events; Redis cache data is evictable, while the jobs Redis has a separate durability role. Do not silently substitute one for another.
- Keep tenant isolation, idempotency, leases, fencing, and audit behavior intact when changing distributed or trading workflows.
- `/api/agent/v1` is the Agent Gateway boundary. Update `docs/agent/agent-openapi.json` and its alignment tests whenever that HTTP contract changes.
- MCP capabilities must retain the same authentication, scopes, idempotency, limits, and live-trading safeguards as the backing API.

## Safety and repository hygiene

- Never commit real secrets, production `.env` files, API keys, or database passwords. Use `.env.example` and placeholders.
- Do not weaken live-trading safeguards or bypass explicit authorization and human review unless the user requests and scopes that change.
- Do not add upgrade-time data rewrites for one-off local cleanup. Use an explicit, reviewed migration only when shipped user data must change.
- Preserve unrelated working-tree changes.
- Use English for source-code comments. Keep user-facing application copy in the existing localization system rather than hardcoding it in source files.
- Keep machine-readable contracts and identifiers in English. When a human guide has English and `_CN` editions, update both when the change affects both audiences.

## Verification

- Run focused backend tests from `backend_api_python/` with `python -m pytest tests/<test_file>.py -q`.
- Agent contract coverage lives in `backend_api_python/tests/test_ai_agent_contract_alignment.py` and related `test_agent_*.py` files.
- MCP tests live in `mcp_server/tests/`.
- Validate Compose changes with `docker compose config --quiet` before exercising the affected services.
- Match verification depth to risk; trading, migrations, tenancy, and distributed ownership require targeted regression tests.

<!-- graft:start -->
## Graft — repo context graph

This repo is indexed in `graft/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `graft map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `graft ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `graft grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `graft skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `graft callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `graft/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `graft ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `graft build` (deterministic,
no API key, $0).
<!-- graft:end -->
