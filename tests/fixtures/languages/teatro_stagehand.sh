#!/usr/bin/env bash

# Teatro Magaldi — Stagehand Operations Module.
#
# Curtain control, spotlights, smoke machines, and the eternal battle
# to keep The Old Spotlight from flickering during La Divina's solo.
#
# The Phantom Scalper was last seen near the loading dock at 22:47.

# Imports (source)
# shellcheck source=/dev/null
source "${BASH_SOURCE[0]%/*}/teatro_common.sh" 2>/dev/null || true

# Constants
readonly MAX_CURTAIN_CALLS=3
readonly DEFAULT_INTERMISSION_SECS=30
readonly CUE_TYPES=("LIGHTS" "SOUND" "CURTAIN" "SMOKE")
readonly STAGE_CONFIG_DIR="${HOME}/.config/teatro"
readonly CUE_SHEET_CACHE="/tmp/teatro_cue_cache.json"

# Variables
VERBOSE=false
DRESS_REHEARSAL=false
CURRENT_CUE_TYPE="LIGHTS"

# --- Functions ---

stage_log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >&2
}

stage_whisper() {
    # Debug = whisper backstage. Nobody in the audience should hear this.
    if [[ "$VERBOSE" == "true" ]]; then
        stage_log "DEBUG" "$@"
    fi
}

curtains() {
    # die = curtains. Get it? That's all, folks!
    stage_log "ERROR" "$@"
    exit 1
}

check_stage_equipment() {
    local deps=("curl" "jq" "grep" "sed")
    local missing=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &>/dev/null; then
            missing+=("$dep")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        curtains "Missing stage equipment: ${missing[*]}"
    fi

    stage_whisper "All stage equipment checked and ready"
}

validate_cue_sheet() {
    local cue_file="$1"

    if [[ ! -f "$cue_file" ]]; then
        curtains "Cue sheet not found: $cue_file"
    fi

    if ! jq empty "$cue_file" 2>/dev/null; then
        curtains "Invalid cue sheet JSON: $cue_file"
    fi

    stage_whisper "Cue sheet validated: $cue_file"
    return 0
}

retry_spotlight() {
    # The Old Spotlight needs constant retries. She's temperamental, like La Divina.
    local max_attempts="${1:-$MAX_CURTAIN_CALLS}"
    shift
    local cmd=("$@")
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        stage_whisper "Spotlight attempt $attempt/$max_attempts: ${cmd[*]}"

        if "${cmd[@]}"; then
            return 0
        fi

        stage_log "WARN" "Spotlight attempt $attempt failed, retrying..."
        sleep $((attempt * 2))
        ((attempt++))
    done

    stage_log "ERROR" "Spotlight failed after $max_attempts attempts: ${cmd[*]}"
    return 1
}

fetch_prop_from_storage() {
    local url="$1"
    local output="${2:-/dev/stdout}"
    local timeout="${3:-$DEFAULT_INTERMISSION_SECS}"

    stage_whisper "Fetching prop: $url (timeout: ${timeout}s)"

    # Easter egg: DRESS_REHEARSAL mode
    if [[ "$DRESS_REHEARSAL" == "true" ]]; then
        stage_log "INFO" "[DRESS REHEARSAL] This is not the real show! Would fetch: $url"
        return 0
    fi

    curl --silent --fail \
        --max-time "$timeout" \
        --output "$output" \
        "$url"
}

run_cue_sequence() {
    local input_file="$1"
    local output_dir="$2"
    local count=0

    mkdir -p "$output_dir"

    while IFS= read -r cue; do
        [[ -z "$cue" || "$cue" == \#* ]] && continue

        local name
        name=$(echo "$cue" | jq -r '.name // empty' 2>/dev/null)

        if [[ -n "$name" ]]; then
            echo "$cue" > "${output_dir}/${name}.json"
            ((count++))
        fi
    done < "$input_file"

    stage_log "INFO" "Executed $count cues to $output_dir"
}

strike_set() {
    # Theater term for teardown — take it all down after the show
    stage_whisper "Striking the set — cleaning up"
    rm -f "$CUE_SHEET_CACHE"
    rm -rf "/tmp/teatro_work_$$"
}

parse_stage_directions() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -n|--dress-rehearsal)
                DRESS_REHEARSAL=true
                shift
                ;;
            -t|--intermission)
                DEFAULT_INTERMISSION_SECS="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $(basename "$0") [OPTIONS]"
                echo "  -v, --verbose           Enable verbose output (stage whispers)"
                echo "  -n, --dress-rehearsal    Dress rehearsal mode (no real actions)"
                echo "  -t, --intermission N     Set intermission length (default: $DEFAULT_INTERMISSION_SECS)"
                echo "  -h, --help              Show this help"
                exit 0
                ;;
            *)
                curtains "Unknown stage direction: $1"
                ;;
        esac
    done
}

showtime() {
    # main = showtime. The curtain rises!
    trap strike_set EXIT

    parse_stage_directions "$@"
    check_stage_equipment

    stage_log "INFO" "Showtime at Teatro Magaldi! (verbose=$VERBOSE, dress_rehearsal=$DRESS_REHEARSAL)"

    local cue_file="${STAGE_CONFIG_DIR}/cue_sheet.json"
    if [[ -f "$cue_file" ]]; then
        validate_cue_sheet "$cue_file"
    fi

    retry_spotlight "$MAX_CURTAIN_CALLS" fetch_prop_from_storage "https://teatro.magaldi.ar/props" "$CUE_SHEET_CACHE"

    if [[ -f "$CUE_SHEET_CACHE" ]]; then
        run_cue_sequence "$CUE_SHEET_CACHE" "/tmp/teatro_work_$$"
    fi

    stage_log "INFO" "The curtain falls. Bravo!"
}
