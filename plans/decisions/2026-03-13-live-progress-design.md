## Decision: Live progress file design for parse monitoring

**Original plan:** Parse logging only writes `_last_run.json` at the END of a run. No mid-run visibility. Agents had to guess progress from PID status, CPU usage, and network connections.

**Deviation:** Added `LiveProgressWriter` that writes `logs/_current_run.json` continuously during parse runs with phase status, progress counts, ETA, and per-type breakdowns.

**Why:** During a 4+ hour parse run, an agent monitoring remotely had zero visibility into progress. Had to resort to checking `ps aux`, counting network connections to Ollama, and estimating based on elapsed time. This is unreliable and uninformative.

**Options considered:**
1. **Append to a log file** — Simple text appending. Pro: easy to implement. Con: agents would need to parse unstructured text, file grows unbounded, harder to extract current state.
2. **Single JSON file with atomic overwrites** — Write entire state to one file, using temp+rename for atomicity. Pro: agents read one file and get complete state, always valid JSON, no parsing ambiguity. Con: slight I/O overhead from full rewrites.
3. **SQLite progress database** — Store progress in SQLite. Pro: queryable, concurrent-safe. Con: overkill for single-writer/single-reader scenario, adds dependency.
4. **Named pipe / socket** — Real-time streaming. Pro: immediate updates. Con: requires agent to connect at the right time, complex error handling, not persistent.

**Final decision:** Option 2 — Single JSON file with atomic overwrites. The file is small (<2KB), written at most every 30 seconds, and provides complete state in a single read. Atomic writes via `tempfile` + `os.replace()` ensure no partial reads. Thread-safe with `threading.Lock`. File is deleted on normal exit and left on crash for PID-based detection. This gives agents everything they need without any guessing.
