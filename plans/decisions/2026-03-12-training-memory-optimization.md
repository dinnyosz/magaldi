## Decision: Reduce MLX training memory footprint

**Original plan:** Train Qwen2.5-1.5B-Instruct with LoRA on all 28 layers, batch_size=4, gradient_accumulation_steps=2

**Deviation:** Removed num_layers config (user edited out), reduced batch_size to 2, increased gradient_accumulation_steps to 4. Also updated token estimation ratio from 4 chars/token to 3 chars/token in generate_scoring_data.py.

**Why:** Each mlx_lm.lora training process was consuming 11GB real memory (10GB in Metal/IOAccelerator GPU buffers) for a 1.5B parameter model. Two orphaned chicane sessions had launched concurrent training runs, totaling 22GB+ unified memory on a 48GB system. The excessive memory came from: LoRA applied to all 28 transformer layers (full optimizer state for each), batch_size=4 keeping 4 sequences of activations in flight simultaneously, and fp16 model weights (~3GB base).

**Options considered:**
1. Reduce num_layers only (28→8) - Cuts optimizer state ~70%, but still high activation memory with batch_size=4
2. Reduce batch_size only (4→1) with higher grad_accum - Cuts peak activation memory ~4x, same effective batch size, but still 28 layers of optimizer state
3. Both changes together - Maximum memory reduction while preserving training dynamics (same effective batch size of 8)
4. Switch to quantized base model (4-bit) - Would cut model weight memory ~4x but mlx-lm LoRA doesn't support QLoRA natively on MLX

**Final decision:** Combined approach — user further refined to batch_size=2, gradient_accumulation_steps=4, and removed num_layers entirely (letting mlx-lm use its default). Also made token estimation more conservative (3 chars/token instead of 4) since code tokenizes less efficiently than prose, ensuring training examples don't silently exceed max_seq_len. Expected per-run memory drop from ~11GB to ~5-6GB.
