# Remote Server Setup (192.168.68.31)

Steps to bring the remote llama.cpp server in line with `config/magaldi.yaml`.

## Current state (2026-11-02)

| Service | Port | State |
|---|---|---|
| OpenSearch | 9200 | Running (indices exist, 0 docs) |
| Redis | 6379 | Running |
| OpenSearch Dashboards | 5601 | Running |
| llama.cpp (Docker) | 8090 | Running, but serves the **wrong models** |
| Web UI | 8080 | **Not running** |

The llama.cpp container serves `Qwen3.5-4B-Q4_K_M`, `Qwen3.5-2B-Q4_K_M` and
`Qwen3-Embedding-0.6B-q8_0` from a hand-written presets ini. The config expects:

| Role | Model |
|---|---|
| `summarize_model`, `variable_scoring_model` | `Qwen3-4B-Instruct-2507-Q4_K_M` (+ draft `Qwen3-0.6B-Q8_0`) |
| `summarize_model_small` | `Qwen3-1.7B-Q4_K_M` |
| `embed_model` | `Qwen3-Embedding-0.6B-q8_0` |

Summarization fails until the server serves these names.

## 1. Update the checkout

```bash
cd <magaldi checkout>
git pull
make setup          # only if .venv does not exist yet
```

## 2. Download the models

```bash
.venv/bin/magaldi llamacpp pull --active
```

Downloads only the models referenced by the role settings into `tools/models/`.
Already-present files are skipped. (`make llama-pull` does the same.)

## 3. Generate the presets ini

```bash
.venv/bin/magaldi llamacpp presets --models-dir /models --active -o <ini path mounted into the container>
```

- `--models-dir /models` is the model directory **as seen inside the container**.
  Every section gets an explicit `model = /models/<file>.gguf` key; context tiers
  (`-1k` ... `-32k`) point at the base GGUF, so no symlinks are needed.
- Omit `-o` to print to stdout and inspect first.
- Re-run this whenever `config/magaldi.yaml` changes the llama.cpp models.

## 4. Update the container

**Volumes**

- Mount `<magaldi checkout>/tools/models` at `/models` (or copy the `.gguf` files
  from step 2 into whatever host dir is currently mounted at `/models`).
- Mount the ini from step 3 and pass it via `--models-preset`.

**Server flags.** The generated ini only holds per-model settings (ctx-size,
parallel, embedding, draft). Server-wide flags must be on the container command,
matching what `magaldi llamacpp serve` uses (`src/shared/cli/llamacpp.py`):

```
llama-server \
  --host 0.0.0.0 --port 8090 \
  --models-preset /path/in/container/models.ini \
  --models-max 2 \
  --flash-attn on \
  --n-gpu-layers 99 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --no-kv-unified \
  -sps 0 \
  --threads <physical cores>
```

**Environment**

```
LLAMA_SET_ROWS=1      # required by --no-kv-unified
```

Do **not** put `--ctx-size`, `--parallel` or `--batch-size` on the command line:
CLI args override the per-model preset values (embedding needs batch-size 12288).

Then restart the container.

## 5. Verify

```bash
curl -s http://192.168.68.31:8090/v1/models \
  | python3 -c 'import json,sys;[print(m["id"]) for m in json.load(sys.stdin)["data"]]'
```

Expected IDs:

- `Qwen3-4B-Instruct-2507-Q4_K_M` and `-1k`, `-2k`, `-4k`, `-8k`, `-16k`, `-32k`
- `Qwen3-1.7B-Q4_K_M` and the same tiers
- `Qwen3-Embedding-0.6B-q8_0`

Smoke test:

```bash
curl -s http://192.168.68.31:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-1.7B-Q4_K_M-1k","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
curl -s http://192.168.68.31:8090/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-Embedding-0.6B-q8_0","input":"hello"}' | head -c 200
```

## 6. Web UI

Not running yet. On the remote:

```bash
docker compose --profile web up -d
curl -sf http://localhost:8080/api/v1/admin/health
```

The web container resolves `opensearch`/`redis` via the compose network. If it
needs to reach llama.cpp on the host, use `host.docker.internal:8090` (add
`extra_hosts: ["host.docker.internal:host-gateway"]` on Linux).

## Security note

OpenSearch (security plugin disabled), Redis (no password) and llama.cpp are
bound to all interfaces with no auth. Fine on a trusted LAN only; otherwise bind
to `127.0.0.1` and use an SSH tunnel, or firewall the ports.

## Still open (Mac side)

- `.env` sets `MAGALDI_ELASTICSEARCH_HOST` / `MAGALDI_REDIS_HOST` to `localhost`;
  change to `192.168.68.31` to use the remote services.
- LLM URLs are hardcoded `http://localhost:8090` in `config/magaldi.yaml`; there is
  no env override yet (planned: `MAGALDI_LLM_URL`).
