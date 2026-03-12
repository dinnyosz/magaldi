## Decision: v2 Training Overfit — Next Steps Strategy

**Original plan:** Train Qwen2.5-1.5B-Instruct for 10 epochs, export to GGUF, evaluate, and ship as production variable scorer.

**Deviation:** The final model (10 epochs, val loss 0.047) produces garbage output — catastrophic overfitting confirmed. The Deep Research section's warning about 10 epochs being too many was correct.

**Why:** Val loss kept decreasing (0.467 → 0.047) which looked promising, but the model memorized training patterns instead of learning to generalize. With only 2538 training examples and 10 full passes, the model overfit to the exact token distributions.

**Options considered:**
1. **Try earlier checkpoints (epoch 2-3)** - Fast, no retraining needed. 63 checkpoints saved every 100 iters. Epoch 2 ≈ iter 1270, Epoch 3 ≈ iter 1905. Pro: immediate testing. Con: still trained with suboptimal hyperparameters (LR 5e-5, rank 16, mask_prompt: true).
2. **Train v3 with optimized config** - 3 epochs, LR 2e-4, rank 8, mask_prompt: false, cosine decay. Pro: addresses all known issues at once. Con: another 3-4 hours of training time.
3. **Try both sequentially** - Test early checkpoints first (quick). If they show promise, ship one. Then train v3 for comparison. Pro: fastest path to a working model. Con: more total time.

**Final decision:** Pending user input. Option 3 (try checkpoints first, then v3) is recommended as it provides the fastest path to a working model while still pursuing the optimized config.

**Evidence:**
- Model output sample: "andy \nInI the variable value value value value value..." (complete gibberish)
- Val loss at epoch 2 (iter 1270): 0.109 — still reasonable, model may not be overfit yet
- User's Deep Research sources unanimously recommend 1-3 epochs for instruction fine-tuning
