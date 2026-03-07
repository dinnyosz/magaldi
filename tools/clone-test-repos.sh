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
# Tier 1 = smoke test (~1 min), Tier 2 = pattern coverage (~5 min)

REPOS=(
  # Python
  "pallets/click|1|python"
  "psf/requests|2|python"
  "encode/httpx|2|python"

  # JavaScript
  "sindresorhus/got|1|javascript"
  "expressjs/express|2|javascript"
  "lodash/lodash|2|javascript"

  # TypeScript
  "colinhacks/zod|1|typescript"
  "trpc/trpc|2|typescript"
  "drizzle-team/drizzle-orm|2|typescript"

  # PHP
  "guzzle/guzzle|1|php"
  "composer/composer|2|php"
  "PHPMailer/PHPMailer|2|php"

  # Rust
  "sharkdp/fd|1|rust"
  "BurntSushi/ripgrep|2|rust"
  "sharkdp/bat|2|rust"

  # Java
  "google/gson|1|java"
  "spring-projects/spring-petclinic|2|java"
  "square/okhttp|2|java"

  # Bash
  "dylanaraps/neofetch|1|bash"
  "rbenv/rbenv|2|bash"
  "nvm-sh/nvm|2|bash"

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
      echo "  --tier 2       All repos (tier 1 + 2, ~150 MB total)"
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
