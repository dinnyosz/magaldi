#!/usr/bin/env bash

# Example Bash file exercising all extractable element types.
#
# This fixture ensures the magaldi index always contains Bash-specific
# elements (function, constant, variable) when parsing its own repo.

# Imports (source)
# shellcheck source=/dev/null
source "${BASH_SOURCE[0]%/*}/common.sh" 2>/dev/null || true

# Constants
readonly MAX_RETRIES=3
readonly DEFAULT_TIMEOUT=30
readonly LOG_LEVELS=("DEBUG" "INFO" "WARN" "ERROR")
readonly CONFIG_DIR="${HOME}/.config/example"
readonly CACHE_FILE="/tmp/example_cache.json"

# Variables
VERBOSE=false
DRY_RUN=false
CURRENT_LOG_LEVEL="INFO"

# --- Functions ---

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >&2
}

debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        log "DEBUG" "$@"
    fi
}

die() {
    log "ERROR" "$@"
    exit 1
}

check_dependencies() {
    local deps=("curl" "jq" "grep" "sed")
    local missing=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &>/dev/null; then
            missing+=("$dep")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        die "Missing dependencies: ${missing[*]}"
    fi

    debug "All dependencies satisfied"
}

validate_config() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        die "Config file not found: $config_file"
    fi

    if ! jq empty "$config_file" 2>/dev/null; then
        die "Invalid JSON in config: $config_file"
    fi

    debug "Config validated: $config_file"
    return 0
}

retry_command() {
    local max_attempts="${1:-$MAX_RETRIES}"
    shift
    local cmd=("$@")
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        debug "Attempt $attempt/$max_attempts: ${cmd[*]}"

        if "${cmd[@]}"; then
            return 0
        fi

        log "WARN" "Attempt $attempt failed, retrying..."
        sleep $((attempt * 2))
        ((attempt++))
    done

    log "ERROR" "Command failed after $max_attempts attempts: ${cmd[*]}"
    return 1
}

fetch_url() {
    local url="$1"
    local output="${2:-/dev/stdout}"
    local timeout="${3:-$DEFAULT_TIMEOUT}"

    debug "Fetching: $url (timeout: ${timeout}s)"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would fetch: $url"
        return 0
    fi

    curl --silent --fail \
        --max-time "$timeout" \
        --output "$output" \
        "$url"
}

process_items() {
    local input_file="$1"
    local output_dir="$2"
    local count=0

    mkdir -p "$output_dir"

    while IFS= read -r item; do
        [[ -z "$item" || "$item" == \#* ]] && continue

        local name
        name=$(echo "$item" | jq -r '.name // empty' 2>/dev/null)

        if [[ -n "$name" ]]; then
            echo "$item" > "${output_dir}/${name}.json"
            ((count++))
        fi
    done < "$input_file"

    log "INFO" "Processed $count items to $output_dir"
}

cleanup() {
    debug "Cleaning up temporary files"
    rm -f "$CACHE_FILE"
    rm -rf "/tmp/example_work_$$"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -t|--timeout)
                DEFAULT_TIMEOUT="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $(basename "$0") [OPTIONS]"
                echo "  -v, --verbose    Enable verbose output"
                echo "  -n, --dry-run    Dry run mode"
                echo "  -t, --timeout N  Set timeout (default: $DEFAULT_TIMEOUT)"
                echo "  -h, --help       Show this help"
                exit 0
                ;;
            *)
                die "Unknown option: $1"
                ;;
        esac
    done
}

main() {
    trap cleanup EXIT

    parse_args "$@"
    check_dependencies

    log "INFO" "Starting example process (verbose=$VERBOSE, dry_run=$DRY_RUN)"

    local config_file="${CONFIG_DIR}/settings.json"
    if [[ -f "$config_file" ]]; then
        validate_config "$config_file"
    fi

    retry_command "$MAX_RETRIES" fetch_url "https://example.com/data" "$CACHE_FILE"

    if [[ -f "$CACHE_FILE" ]]; then
        process_items "$CACHE_FILE" "/tmp/example_work_$$"
    fi

    log "INFO" "Done"
}
