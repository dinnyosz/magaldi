#!/usr/bin/env bash
# Parse all cloned test repositories one by one.
# Each repo uses scope=test-repo and repository=<dirname>.
# Ensures magaldi.yaml exists with the right values before parsing.
#
# Usage:
#   ./tools/parse-test-repos.sh                  # Parse all cloned repos
#   ./tools/parse-test-repos.sh --tier 1         # Smoke test only
#   ./tools/parse-test-repos.sh --lang rust      # Single language
#   ./tools/parse-test-repos.sh --dry-run        # In-memory, no DB needed
#   ./tools/parse-test-repos.sh --skip-ai        # Skip summarization/embedding
#   ./tools/parse-test-repos.sh --list           # Just list what would be parsed
#   ./tools/parse-test-repos.sh click ripgrep    # Parse specific repos by name
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEST="$PROJECT_ROOT/test_repos"
SCOPE="test-repo"
USER="test"

# ── Repo definitions (mirrors clone-test-repos.sh) ─────────────────
# Format: "dirname|tier|language"

REPOS=(
  "click|1|python"
  "requests|2|python"
  "httpx|2|python"
  "got|1|javascript"
  "express|2|javascript"
  "lodash|2|javascript"
  "zod|1|typescript"
  "trpc|2|typescript"
  "drizzle-orm|2|typescript"
  "guzzle|1|php"
  "composer|2|php"
  "PHPMailer|2|php"
  "fd|1|rust"
  "ripgrep|2|rust"
  "bat|2|rust"
  "neofetch|1|bash"
  "rbenv|2|bash"
  "nvm|2|bash"
  "nickel|2|polyglot"
)

# ── Argument parsing ────────────────────────────────────────────────

TIER_FILTER=""
LANG_FILTER=""
LIST_ONLY=false
EXTRA_ARGS=()
SPECIFIC_REPOS=()

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
    --dry-run|--skip-ai|--skip-tests|--skip-resolve|--force-clean)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --workers|-w)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --llm-url)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS] [REPO_NAME...]"
      echo ""
      echo "Options:"
      echo "  --tier 1|2       Filter by tier (1=smoke, 2=all)"
      echo "  --lang LANG      Filter by language"
      echo "  --list           Print repos without parsing"
      echo "  --dry-run        Use in-memory storage (no DB needed)"
      echo "  --skip-ai        Skip summarization and embedding"
      echo "  --skip-tests     Exclude test directories"
      echo "  --skip-resolve   Skip call resolution"
      echo "  --force-clean    Delete existing index data before parsing"
      echo "  --workers N      Max parallel workers"
      echo "  --llm-url URL    Override LLM API URL"
      echo ""
      echo "Positional args are repo directory names to parse (e.g., click ripgrep)."
      echo "If none given, all matching repos are parsed."
      echo ""
      echo "Repos are expected in: $DEST"
      echo "Run ./tools/clone-test-repos.sh first to clone them."
      exit 0
      ;;
    -*)
      echo "Unknown option: $1 (try --help)"
      exit 1
      ;;
    *)
      # Positional arg = specific repo name
      SPECIFIC_REPOS+=("$1")
      shift
      ;;
  esac
done

# ── Ensure magaldi.yaml ────────────────────────────────────────────

ensure_config() {
  local repo_dir="$1"
  local name="$2"
  local config_file="$repo_dir/magaldi.yaml"

  # Always overwrite to ensure scope=test-repo
  cat > "$config_file" <<EOF
scope: $SCOPE
repository: $name
EOF
}

# ── Main ───────────────────────────────────────────────────────────

if [[ ! -d "$DEST" ]]; then
  echo "ERROR: test_repos/ directory not found."
  echo "Run ./tools/clone-test-repos.sh first."
  exit 1
fi

MATCHING=()
for entry in "${REPOS[@]}"; do
  IFS='|' read -r name tier lang <<< "$entry"

  # Filter by specific repos
  if [[ ${#SPECIFIC_REPOS[@]} -gt 0 ]]; then
    found=false
    for specific in "${SPECIFIC_REPOS[@]}"; do
      [[ "$name" == "$specific" ]] && found=true && break
    done
    $found || continue
  fi

  # Apply tier/lang filters
  [[ -n "$TIER_FILTER" && "$tier" -gt "$TIER_FILTER" ]] && continue
  [[ -n "$LANG_FILTER" && "$lang" != "$LANG_FILTER" ]] && continue

  MATCHING+=("$entry")
done

if [[ ${#MATCHING[@]} -eq 0 ]]; then
  echo "No repos match the given filters."
  exit 0
fi

if $LIST_ONLY; then
  printf "\n%-20s  %-6s  %-12s  %s\n" "REPO" "TIER" "LANGUAGE" "STATUS"
  printf "%-20s  %-6s  %-12s  %s\n" "────" "────" "────────" "──────"
  for entry in "${MATCHING[@]}"; do
    IFS='|' read -r name tier lang <<< "$entry"
    if [[ -d "$DEST/$name" ]]; then
      status="cloned"
    else
      status="NOT CLONED"
    fi
    printf "%-20s  %-6s  %-12s  %s\n" "$name" "$tier" "$lang" "$status"
  done
  echo ""
  exit 0
fi

total=${#MATCHING[@]}
passed=0
failed=0
skipped=0
failed_repos=()

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Magaldi Test Repo Parser                                   ║"
echo "║  Scope: $SCOPE                                         ║"
echo "║  Repos: $total                                              ║"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
echo "║  Flags: ${EXTRA_ARGS[*]}"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

start_time=$SECONDS

for i in "${!MATCHING[@]}"; do
  entry="${MATCHING[$i]}"
  IFS='|' read -r name tier lang <<< "$entry"
  idx=$((i + 1))

  repo_dir="$DEST/$name"

  if [[ ! -d "$repo_dir" ]]; then
    echo "[$idx/$total] SKIP $name (not cloned)"
    ((skipped++))
    continue
  fi

  echo "────────────────────────────────────────────────────────────"
  echo "[$idx/$total] PARSE $name ($lang, tier $tier)"
  echo "────────────────────────────────────────────────────────────"

  # Ensure magaldi.yaml has the right scope/repo
  ensure_config "$repo_dir" "$name"

  repo_start=$SECONDS

  if magaldi parse "$repo_dir" --user "$USER" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
    elapsed=$((SECONDS - repo_start))
    echo ""
    echo "[$idx/$total] PASS $name (${elapsed}s)"
    ((passed++))
  else
    elapsed=$((SECONDS - repo_start))
    echo ""
    echo "[$idx/$total] FAIL $name (${elapsed}s)"
    ((failed++))
    failed_repos+=("$name")
  fi

  echo ""
done

total_time=$((SECONDS - start_time))

echo "════════════════════════════════════════════════════════════════"
echo "  RESULTS: $passed passed, $failed failed, $skipped skipped ($total_time seconds)"
if [[ ${#failed_repos[@]} -gt 0 ]]; then
  echo "  FAILED:  ${failed_repos[*]}"
fi
echo "════════════════════════════════════════════════════════════════"

# Exit with failure if any repo failed
[[ $failed -eq 0 ]]
