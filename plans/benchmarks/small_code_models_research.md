# Small Language Models for Code Summarization: Research Analysis

> **Date**: January 2026
> **Purpose**: Identify state-of-the-art small models (1-10B parameters) for code summarization tasks

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Code Summarization Benchmarks](#code-summarization-benchmarks)
3. [Small Model Comparison](#small-model-comparison)
4. [Detailed Model Analysis](#detailed-model-analysis)
5. [Recommendations for Magaldi](#recommendations-for-magaldi)
6. [Sources](#sources)

---

## Executive Summary

### Key Findings

1. **Best Overall Small Code Models** (by efficiency/performance ratio):
   - **Qwen2.5-Coder-1.5B/3B**: Best balance of size, speed, and code understanding
   - **DeepSeek-Coder-1.3B/6.7B**: Strong code explanation capabilities
   - **Granite-3B-Code**: Best small model for code explanation tasks

2. **Code Summarization vs Code Generation**:
   - Most benchmarks focus on code generation (HumanEval, MBPP)
   - Dedicated code summarization benchmarks are limited (CodeXGLUE, HumanEvalExplain)
   - **HumanEvalExplain** is the primary benchmark for code explanation/summarization

3. **Critical Insight**: IBM Granite models significantly outperform other models on **HumanEvalExplain** (code explanation), making them particularly suitable for summarization tasks.

---

## Code Summarization Benchmarks

### Primary Benchmarks

| Benchmark | Task | Metrics | Languages |
|-----------|------|---------|-----------|
| **HumanEvalExplain** | Generate explanation of code | Pass@1 (functional correctness) | Python, multilingual |
| **CodeXGLUE** | Code-to-text summarization | Smoothed BLEU | Python, Java, PHP, JS, Ruby, Go |
| **CodeSearchNet** | Generate docstrings from code | BLEU, BERTScore, ROUGE | 6 languages |

### Evaluation Metrics

| Metric | Description | Correlation with Human Judgment |
|--------|-------------|--------------------------------|
| **BLEU** | N-gram exact match | Moderate (~0.4-0.5 Spearman) |
| **BERTScore** | Semantic embedding similarity | Better (~0.5-0.6 Spearman) |
| **ROUGE-L** | Longest common subsequence | Moderate |
| **CODERPE** (LLM-based) | LLM evaluator | Best (0.82 Spearman) |

> **Note**: Research shows BERTScore and BLEU are highly correlated, providing little additional insight when used together. LLM-based evaluation (using another model as judge) correlates best with human judgment.

---

## Small Model Comparison

### Code Generation Benchmarks (HumanEval/MBPP)

| Model | Size | HumanEval | MBPP | VRAM | License |
|-------|------|-----------|------|------|---------|
| **Qwen2.5-Coder-1.5B** | 1.5B | 43.9% | 69.2% | ~6.5 GB | Apache 2.0 |
| **Qwen2.5-Coder-3B** | 3B | 52.4% | 72.2% | ~10.8 GB | Qwen-Research |
| **DeepSeek-Coder-1.3B** | 1.3B | ~45% | ~65% | ~5 GB | Apache 2.0 |
| **DeepSeek-Coder-6.7B** | 6.7B | 56% | 70% | ~14 GB | Apache 2.0 |
| **Yi-Coder-1.5B** | 1.5B | 41.5% | ~60% | ~6 GB | Apache 2.0 |
| **Yi-Coder-9B** | 9B | 85.4% | 73.8% | ~20 GB | Apache 2.0 |
| **StarCoder2-3B** | 3B | ~45% | ~55% | ~6 GB | BigCode-OS |
| **StarCoder2-7B** | 7B | ~50% | ~60% | ~14 GB | BigCode-OS |
| **CodeGemma-2B** | 2B | ~35% | ~50% | ~4.4 GB | Gemma |
| **CodeGemma-7B** | 7B | ~50% | ~60% | ~14 GB | Gemma |
| **Phi-3-mini** | 3.8B | 58.5% | 70.0% | ~8 GB | MIT |
| **Granite-3B-Code** | 3B | ~50% | ~65% | ~6 GB | Apache 2.0 |
| **Granite-8B-Code** | 8B | ~55% | ~70% | ~16 GB | Apache 2.0 |

### Code Explanation/Summarization (HumanEvalExplain)

| Model | Size | Python | Avg (Multi-lang) | Notes |
|-------|------|--------|------------------|-------|
| **MagiCoder-DS-6.7B** | 6.7B | **55.5%** | - | Best small model |
| **DeepSeek-Coder-6.7B** | 6.7B | 43.9% | 34.6% | Strong performer |
| **Llama3-8B-Instruct** | 8B | 42.7% | - | General purpose |
| **WaveCoder-CL-7B** | 7B | 41.4% | - | Code-focused |
| **Granite-8B-Code-Base** | 8B | ~52% | ~45% | Beats CodeLlama-34B |
| **Granite-3B-Code-Base** | 3B | ~45% | ~38% | Best-in-class for size |
| **CodeLlama-7B** | 7B | 33.5% | - | Baseline |

### CodeXGLUE Summarization (BLEU Scores)

| Model | Size | BLEU Score | Notes |
|-------|------|------------|-------|
| **CodeLlama-7B** | 7B | 20.39 | Baseline |
| **CodeLlama-13B** | 13B | 21.15 | Larger model |
| **CodeT5** | 220M | ~19 | Encoder-decoder |
| **CodeBERT** | 125M | ~17 | Understanding focused |

---

## Detailed Model Analysis

### 1. Qwen2.5-Coder Family

**Strengths:**
- State-of-the-art performance across most code benchmarks
- Excellent efficiency-to-performance ratio
- Strong multilingual support (40+ languages)
- Active development with frequent updates

**Sizes Available:** 0.5B, 1.5B, 3B, 7B, 14B, 32B

**Best For Summarization:**
- **1.5B**: Fast inference, good for simple functions/methods
- **3B**: Better quality, suitable for classes and files

**Recommended Settings** (from HuggingFace):
```
temperature: 0.7
top_p: 0.8
top_k: 20
min_p: 0.0
presence_penalty: 1.5 (non-thinking mode)
```

### 2. IBM Granite Code Models

**Strengths:**
- **Significantly outperforms on HumanEvalExplain** (code explanation)
- Granite-8B-Code-Base beats CodeLlama-34B by 9.3% on code explanation
- Strong at code fixing and repair tasks
- Enterprise-focused with good documentation

**Sizes Available:** 3B, 8B, 20B, 34B

**Best For Summarization:**
- **3B**: Best small model for code explanation tasks
- **8B**: Near CodeLlama-70B performance on explanation

**Key Insight**: Granite models are specifically optimized for code understanding (not just generation), making them ideal for summarization tasks.

### 3. DeepSeek-Coder

**Strengths:**
- Excellent code generation performance
- 6.7B model performs on par with CodeLlama-34B
- Strong on HumanEvalExplain

**Sizes Available:** 1.3B, 6.7B, 33B

**Best For Summarization:**
- **1.3B**: Highest METEOR score (44.08) on code summarization
- **6.7B**: Best BLEURT (0.568) and BLEU4 (11.89) scores

### 4. Yi-Coder

**Strengths:**
- 9B model achieves 85.4% on HumanEval (best under 10B)
- First open-source code LLM to exceed 50% on CRUXEval-O
- 128K context window for project-level understanding

**Sizes Available:** 1.5B, 9B

**Best For Summarization:**
- **9B**: Best for complex, multi-file understanding

### 5. StarCoder2

**Strengths:**
- Trained on 600+ programming languages
- Strong code completion capabilities
- 3B model outperforms StarCoderBase-15B

**Sizes Available:** 3B, 7B, 15B

**Best For Summarization:**
- **3B**: Good baseline, fast inference

### 6. CodeGemma

**Strengths:**
- 2B model extremely fast (critical for IDE usage)
- Strong code completion
- Good balance of speed and quality

**Sizes Available:** 2B, 7B

**Best For Summarization:**
- **2B**: When speed is critical
- **7B**: Better quality, used in academic research

### 7. Phi-3-mini

**Strengths:**
- 3.8B parameters rivals Mixtral 8x7B and GPT-3.5
- Strong general reasoning
- Can run on phones

**Best For Summarization:**
- Good general-purpose option when not using code-specific models

---

## Recommendations for Magaldi

### Tier 1: Recommended Models for Code Summarization

| Model | Use Case | Rationale |
|-------|----------|-----------|
| **Qwen2.5-Coder-3B** | Primary summarizer | Best balance of quality and speed |
| **Qwen2.5-Coder-1.5B** | Fast summarizer | When speed is critical |
| **Granite-3B-Code-Instruct** | Explanation tasks | Best-in-class for code explanation |

### Tier 2: Strong Alternatives

| Model | Use Case | Rationale |
|-------|----------|-----------|
| **DeepSeek-Coder-6.7B** | Higher quality | Strong HumanEvalExplain scores |
| **Yi-Coder-9B** | Complex codebases | 128K context, excellent reasoning |
| **Qwen2.5-Coder-7B** | Quality-focused | When resources allow |

### Tier 3: Specialized/Experimental

| Model | Use Case | Rationale |
|-------|----------|-----------|
| **MagiCoder-DS-6.7B** | Best explanation | Highest HumanEvalExplain (55.5%) |
| **CodeGemma-2B** | Ultra-fast | IDE/real-time applications |

### Models to Avoid for Summarization

| Model | Reason |
|-------|--------|
| **LFM2.5** | Explicitly not recommended for coding tasks |
| **PolyCoder** | Very low benchmark scores (1-3%) |
| **Older StarCoder (v1)** | Superseded by StarCoder2 |

### Recommended Benchmark Configuration

```python
models: list[str] = [
    "qwen2.5-coder:1.5b",     # Fast, good quality
    "qwen2.5-coder:3b",       # Best balance
    "granite3.1-moe:3b",      # Strong explanation
    "deepseek-coder:1.3b",    # Fast alternative
]

eval_models: list[str] = [
    "qwen2.5-coder:7b",       # Quality evaluator
    "granite3.1-moe:3b",      # Fast evaluator
]
```

---

## Key Takeaways

1. **Qwen2.5-Coder dominates** most code benchmarks and should be the primary model family for Magaldi.

2. **Granite models excel at code explanation** - consider adding them for summarization tasks specifically.

3. **HumanEvalExplain** is the most relevant benchmark for code summarization (vs HumanEval which tests generation).

4. **LLM-as-judge evaluation** (what Magaldi uses) correlates better with human judgment than BLEU/BERTScore.

5. **3B parameter models** offer the best tradeoff for edge deployment with acceptable quality.

6. **Context length matters** - Yi-Coder's 128K context enables better project-level understanding.

---

## Sources

### Academic Papers
- [Assessing Small Language Models for Code Generation](https://arxiv.org/abs/2507.03160) - Comprehensive SLM evaluation (2025)
- [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)
- [Large Language Models for Code Summarization](https://arxiv.org/html/2405.19032)
- [Calibration of LLMs on Code Summarization](https://arxiv.org/pdf/2404.19318) (FSE 2025)
- [IBM Granite Code Models](https://arxiv.org/pdf/2405.04324)
- [StarCoder 2 and The Stack v2](https://arxiv.org/abs/2402.19173)
- [CodeXGLUE Benchmark](https://arxiv.org/abs/2102.04664)

### Leaderboards & Benchmarks
- [BigCodeBench Leaderboard](https://bigcode-bench.github.io/)
- [HuggingFace BigCode Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)
- [CodeXGLUE](https://microsoft.github.io/CodeXGLUE/)
- [Papers With Code - Code Summarization](https://paperswithcode.com/task/code-summarization)

### Model Documentation
- [Qwen2.5-Coder Family Blog](https://qwenlm.github.io/blog/qwen2.5-coder-family/)
- [Yi-Coder GitHub](https://github.com/01-ai/Yi-Coder)
- [DeepSeek-Coder GitHub](https://github.com/deepseek-ai/DeepSeek-Coder)
- [IBM Granite Code Models](https://github.com/ibm-granite/granite-code-models)
- [StarCoder2 GitHub](https://github.com/bigcode-project/starcoder2)
- [CodeGemma Model Card](https://ai.google.dev/gemma/docs/codegemma/model_card)

### Ollama Models
- [qwen2.5-coder](https://ollama.com/library/qwen2.5-coder)
- [deepseek-coder](https://ollama.com/library/deepseek-coder)
- [starcoder2](https://ollama.com/library/starcoder2)
- [codegemma](https://ollama.com/library/codegemma)
- [granite3.1-moe](https://ollama.com/library/granite3.1-moe)

---

*Last updated: January 2026*
