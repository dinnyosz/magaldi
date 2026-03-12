## Decision: Protect best-known level during GSS bracket narrowing

**Original plan:** GSS narrowing is one-directional — bracket only shrinks via golden-ratio comparisons. Once narrowed past a level, it's excluded permanently.

**Deviation:** Added bracket expansion (Step 0) before narrowing, and clamped narrowing boundaries to never exclude the best-known level.

**Why:** Real-world data showed GSS consistently narrowing past the actual optimal level. Level 1 (0.9s base_time) was excluded when early warmup data at levels 8 and 21 caused the bracket to narrow to [8,20]. The golden-ratio comparison between m1~8 and m2~11 discarded the lower half [1,8] because level 11 appeared better than level 8 — but the actual best (level 1) was too far from either probe point to influence the decision.

**Options considered:**
1. **Keep one-way narrowing, improve probe point selection** — Would still miss cases where early data is misleading. Doesn't solve the fundamental problem of permanent exclusion.
2. **Add bracket expansion + narrowing protection (chosen)** — Bracket expands to include best-known level, narrowing clamped to never exclude it. Simple, robust, works for non-unimodal curves.
3. **Restart GSS when better data found outside bracket** — More disruptive, throws away narrowing progress. Expansion + protection is more incremental.

**Final decision:** Option 2. The bracket now always contains the best-known level. This effectively makes GSS a "converge around the best" system rather than pure golden-section search, but that matches the actual goal — find the optimal concurrency level, not mathematically bisect a unimodal function.

## Decision: Re-evaluate peak after GSS convergence

**Original plan:** GSS `best_level` is set once by `_finalize()` and used as `peak_concurrency` forever after.

**Deviation:** After convergence, every throttle cycle checks if any qualified level has a better base_time than `best_level` and updates it.

**Why:** Screenshot showed Peak@12 while level 1 (0.9s) was clearly best. GSS converged on level 12 based on early data, but as more data accumulated, level 1 proved better. The frozen peak prevented the system from adapting.

**Options considered:**
1. **Keep frozen peak** — Simple but wrong when early convergence picks a suboptimal level.
2. **Re-evaluate from qualified level_data each cycle (chosen)** — Minimal overhead (one `min()` call), self-correcting, logs changes for visibility.
3. **Restart GSS entirely when peak looks wrong** — Overkill, causes instability.

**Final decision:** Option 2. Simple `min()` check on `level_data` each cycle. Peak updates are logged as `PEAK UPDATE: old→new` for observability.
