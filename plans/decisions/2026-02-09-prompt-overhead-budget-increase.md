## Decision: Increase PROMPT_OVERHEAD output budgets based on empirical data

**Original plan:** The context tier accuracy metrics plan (Plan 2 in the plan file) called for adding metrics to detect misestimation — it did NOT prescribe changing the budgets themselves. The original PROMPT_OVERHEAD values were estimated theoretically (system prompt + user template + variable content + output budget) without empirical validation.

**Deviation:** After the tier accuracy metrics revealed massive mismatches, we increased all output budgets in PROMPT_OVERHEAD based on empirical data from a full parse of the magaldi codebase (~2300 elements). This causes significant tier redistribution — methods and functions can no longer fit in the 1024 tier.

**Why:** Empirical data showed:
- method@1024: 197 overflows (8.5% of all methods), worst headroom -30%
- method@2048: 4 overflows (prompts 2162-2216)
- ALL output types exceeded their budgets (variable: 356 max vs 100 budget, file: 405 vs 200, function: 396 vs 200)

The overflows mean the LLM receives a context window that's too small for the actual prompt, potentially causing truncation and degraded summaries.

**Options considered:**
1. **Increase output budgets** — Set budgets to max observed * 1.25. Pros: eliminates overflows, matches real LLM behavior. Cons: bumps many elements to higher tiers (more memory, slightly slower processing due to larger KV cache).
2. **Lower summarize_max_tokens** — Cap LLM output more aggressively. Pros: keeps elements in smaller tiers. Cons: artificially truncates summaries, loses information, doesn't fix input overflow.
3. **Increase variable content estimates only** — Keep output budgets the same but increase the "variable content" portion. Pros: targeted fix for input overflow. Cons: doesn't address the output budget exceedances which are the bigger problem.
4. **Do nothing, treat as informational** — Use metrics for monitoring only. Pros: no code change risk. Cons: overflows continue causing potential summary quality issues.

**Final decision:** Option 1 — increase output budgets. The empirical data is clear: the theoretical estimates were systematically too low across all types. Setting budgets to max * 1.25 gives safety margin while matching real behavior. The tier redistribution (methods/functions leaving 1024 tier) is actually correct — they were overflowing there anyway. The tradeoff of slightly higher memory usage is worth eliminating overflow-induced quality degradation.

Key changes:
| Type | Old Output Budget | New Output Budget | Old Overhead | New Overhead |
|------|------------------|------------------|-------------|-------------|
| file | 200 | 500 | 650 | 950 |
| class | 200 | 450 | 750 | 1000 |
| function | 200 | 500 | 800 | 1100 |
| method | 150 | 400 | 800 | 1050 |
| interface | 150 | 300 | 550 | 700 |
| trait | 150 | 300 | 550 | 700 |
| enum | 150 | 250 | 450 | 600 |
| type_alias | 100 | 200 | 450 | 550 |
| constant | 100 | 200 | 550 | 650 |
| variable | 100 | 450 | 550 | 900 |
| import | 50 | 100 | 350 | 400 |
