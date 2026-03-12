# llama.cpp (Build Dependency)

llama.cpp is used by the training pipeline to convert fine-tuned models to GGUF format for Ollama serving. It is not checked into the repo.

## Install

```bash
git clone https://github.com/ggml-org/llama.cpp tools/llama.cpp
cmake -B tools/llama.cpp/build tools/llama.cpp
cmake --build tools/llama.cpp/build --config Release -j $(sysctl -n hw.ncpu)
```

On Apple Silicon this builds with Metal acceleration automatically.

## Verify

```bash
ls tools/llama.cpp/convert_hf_to_gguf.py      # HF → GGUF converter
ls tools/llama.cpp/build/bin/llama-quantize     # Quantizer (Q4_K_M, Q8_0, etc.)
```

## Usage

The `export_to_ollama.py` script uses these tools via `--llama-cpp-path`:

```bash
python tools/training/export_to_ollama.py \
  --model-dir tools/training/models/variable-scorer-v3/merged \
  --llama-cpp-path tools/llama.cpp \
  --quantization Q4_K_M \
  --ollama-name magaldi-scorer
```

## Update

```bash
cd tools/llama.cpp && git pull && cmake --build build --config Release -j $(sysctl -n hw.ncpu)
```
