## Decision: Align 0.5B and 1.7B training configs for fair comparison

**Original plan:** The 0.5B config had different hyperparameters (rank 16, alpha 32, dropout 0.1, 5 epochs, batch_size 2, mask_prompt true, max_seq_len 2048) vs the 1.7B config (rank 8, alpha 16, dropout 0.05, 3 epochs, batch_size 4, mask_prompt false, max_seq_len 1024).

**Deviation:** Aligned all hyperparameters to match the 1.7B config. Only `base_model`, `ollama_model_name`, `batch_size` (2 vs 4), and `gradient_accumulation_steps` (4 vs 2) differ.

**Why:** Different hyperparameters make it impossible to fairly compare model sizes. Higher LoRA rank (16) on a smaller model risks overfitting. Using identical training settings isolates model size as the only variable.

**Options considered:**
1. Keep separate hyperparameters per model size — allows per-model tuning but confounds comparison
2. Align all hyperparameters (chosen) — fair comparison, can diverge later once baseline is established
3. Create a shared base config with per-model overrides — cleaner but adds complexity prematurely

**Final decision:** Option 2. Align now for a clean baseline comparison, then tune per-model if needed. The 0.5B keeps smaller batch_size (2) with higher gradient_accumulation_steps (4) to maintain the same effective batch size of 8 while fitting in memory.
