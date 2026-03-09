#!/usr/bin/env bash
# Clone external test repositories for parser testing.
# All repos are shallow-cloned into test_repos/ (gitignored).
# Each gets a magaldi.yaml so you can immediately `magaldi parse` them.
#
# Usage:
#   ./tools/clone-test-repos.sh              # Clone all repos
#   ./tools/clone-test-repos.sh --tier 1     # Smoke test only (smallest)
#   ./tools/clone-test-repos.sh --tier 2     # Tier 1 + pattern coverage
#   ./tools/clone-test-repos.sh --lang rust  # Single language
#   ./tools/clone-test-repos.sh --list       # Just print what would be cloned
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEST="$PROJECT_ROOT/test_repos"

# ── Repo definitions ────────────────────────────────────────────────
# Format: "github_path|tier|language"
# Tier 1 = smoke test (~1 min), Tier 2 = pattern coverage (~5 min), Tier 3 = stress test (~15 min)

REPOS=(
  # Python
  "pallets/click|1|python"
  "psf/requests|2|python"
  "encode/httpx|2|python"
  "fastapi/fastapi|3|python"                         # Library: decorators, type hints, async
  "pydantic/pydantic|3|python"                       # Library: dataclasses, generics, validators
  "tiangolo/full-stack-fastapi-template|3|python"    # App using FastAPI/Pydantic/SQLModel
  "home-assistant/core|3|python"                     # App: massive real-world, async, decorators

  # JavaScript
  "sindresorhus/got|1|javascript"
  "expressjs/express|2|javascript"
  "lodash/lodash|2|javascript"
  "axios/axios|3|javascript"                         # Library: CommonJS + ESM patterns
  "date-fns/date-fns|3|javascript"                   # Library: many small modules
  "TryGhost/Ghost|3|javascript"                      # App using Express, knex, etc.
  "goldbergyoni/nodebestpractices|3|javascript"       # App: example code + patterns

  # TypeScript
  "colinhacks/zod|1|typescript"
  "trpc/trpc|2|typescript"
  "drizzle-team/drizzle-orm|2|typescript"
  "prisma/prisma|3|typescript"                       # Library: complex monorepo
  "typeorm/typeorm|3|typescript"                     # Library: decorators, ORM patterns
  "calcom/cal.com|3|typescript"                      # App: Next.js + tRPC + Prisma + zod
  "immich-app/immich|3|typescript"                   # App: NestJS backend + React frontend

  # PHP
  "guzzle/guzzle|1|php"
  "composer/composer|2|php"
  "PHPMailer/PHPMailer|2|php"
  "laravel/framework|3|php"                          # Library: all PHP patterns
  "symfony/console|3|php"                            # Library: CLI component
  "firefly-iii/firefly-iii|3|php"                    # App: Laravel finance tracker
  "matomo-org/matomo|3|php"                          # App: analytics platform

  # Rust
  "sharkdp/fd|1|rust"
  "BurntSushi/ripgrep|2|rust"
  "sharkdp/bat|2|rust"
  "tokio-rs/tokio|3|rust"                            # Library: async runtime, macros
  "serde-rs/serde|3|rust"                            # Library: proc macros, derives
  "astral-sh/ruff|3|rust"                            # App using tokio, serde, clap
  "zellij-org/zellij|3|rust"                         # App: terminal, async, traits, enums

  # Java
  "google/gson|1|java"
  "spring-projects/spring-petclinic|2|java"
  "square/okhttp|2|java"
  "apache/commons-lang|3|java"                       # Library: utilities, generics
  "junit-team/junit5|3|java"                         # Library: annotations, nested classes
  "iluwatar/java-design-patterns|3|java"             # App: all OOP patterns, inner classes
  "apache/kafka|3|java"                              # App: concurrency, annotations

  # Bash
  "dylanaraps/neofetch|1|bash"
  "rbenv/rbenv|2|bash"
  "nvm-sh/nvm|2|bash"
  "ohmyzsh/ohmyzsh|3|bash"                           # Library: many scripts, plugin system
  "asdf-vm/asdf|3|bash"                              # Library: plugin-based version manager
  "pi-hole/pi-hole|3|bash"                           # App: infra scripts, function chains
  "dokku/dokku|3|bash"                               # App: plugin-based PaaS

  # Polyglot
  "nickel-lang/nickel|2|polyglot"
)

# ── Argument parsing ────────────────────────────────────────────────

TIER_FILTER=""
LANG_FILTER=""
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)
      TIER_FILTER="$2"
      shift 2
      ;;
    --lang|--language)
      LANG_FILTER="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
      shift 2
      ;;
    --list)
      LIST_ONLY=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--tier 1|2] [--lang LANGUAGE] [--list]"
      echo ""
      echo "Options:"
      echo "  --tier 1       Smoke test repos only (smallest, ~20 MB total)"
      echo "  --tier 2       Tier 1 + pattern coverage (~150 MB total)"
      echo "  --tier 3       All repos incl. stress tests (~500 MB total)"
      echo "  --lang LANG    Single language: python, javascript, typescript, php, rust, java, bash, polyglot"
      echo "  --list         Print repos without cloning"
      echo ""
      echo "Repos are cloned into: $DEST"
      exit 0
      ;;
    *)
      echo "Unknown option: $1 (try --help)"
      exit 1
      ;;
  esac
done

# ── Filter and clone ────────────────────────────────────────────────

mkdir -p "$DEST"

cloned=0
skipped=0
failed=0

for entry in "${REPOS[@]}"; do
  IFS='|' read -r repo tier lang <<< "$entry"
  name="${repo##*/}"

  # Apply filters
  if [[ -n "$TIER_FILTER" && "$tier" -gt "$TIER_FILTER" ]]; then
    continue
  fi
  if [[ -n "$LANG_FILTER" && "$lang" != "$LANG_FILTER" ]]; then
    continue
  fi

  if $LIST_ONLY; then
    printf "  %-30s  tier=%s  lang=%s\n" "$repo" "$tier" "$lang"
    continue
  fi

  if [ -d "$DEST/$name" ]; then
    echo "SKIP  $name (already exists)"
    ((skipped++))
    continue
  fi

  echo "CLONE $repo → test_repos/$name"
  if git clone --depth 1 --quiet "https://github.com/$repo.git" "$DEST/$name" 2>/dev/null; then
    # Auto-generate magaldi.yaml
    cat > "$DEST/$name/magaldi.yaml" <<EOF
scope: test
repository: $name
EOF
    ((cloned++))
  else
    echo "FAIL  $repo — clone failed"
    ((failed++))
  fi
done

if $LIST_ONLY; then
  exit 0
fi

echo ""
echo "Done: $cloned cloned, $skipped already existed, $failed failed"
echo "Location: $DEST"
echo ""
echo "Parse a repo:  magaldi parse test_repos/<name> --user test"
echo "Parse all:     for d in test_repos/*/; do magaldi parse \"\$d\" --user test; done"
