#!/usr/bin/env python3
"""Export a fine-tuned model to GGUF and register with Ollama.

Steps:
1. Convert HF Safetensors → GGUF (via llama.cpp or mlx-lm)
2. Quantize to Q4_K_M
3. Generate Modelfile
4. Register with `ollama create`
5. Run a quick smoke test

Usage:
    python tools/training/export_to_ollama.py \
      --model-dir tools/training/models/variable-scorer-v1/merged \
      --output-dir tools/training/exports/ \
      --quantization Q4_K_M \
      --ollama-name magaldi-scorer \
      --llama-cpp-path ~/llama.cpp \
      --test
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_to_gguf(
    model_dir: str,
    output_dir: str,
    quantization: str,
    llama_cpp_path: str | None = None,
) -> Path:
    """Convert HuggingFace model to quantized GGUF.

    Tries llama.cpp first (if path provided), falls back to mlx-lm convert.

    Returns path to the GGUF file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gguf_name = f"magaldi-variable-scorer-{quantization.lower()}.gguf"
    gguf_path = output_path / gguf_name

    if llama_cpp_path:
        # Method 1: llama.cpp convert_hf_to_gguf.py
        convert_script = Path(llama_cpp_path) / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            logger.error("convert_hf_to_gguf.py not found at %s", convert_script)
            sys.exit(1)

        # First convert to f16 GGUF
        f16_path = output_path / "model-f16.gguf"
        cmd = [
            sys.executable,
            str(convert_script),
            model_dir,
            "--outtype", "f16",
            "--outfile", str(f16_path),
        ]
        logger.info("Converting to GGUF (f16): %s", " ".join(cmd))
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("GGUF conversion failed: %s", result.stderr)
            sys.exit(1)

        # Then quantize
        quantize_bin = Path(llama_cpp_path) / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            # Try alternative path
            quantize_bin = Path(llama_cpp_path) / "llama-quantize"

        if quantize_bin.exists():
            cmd = [str(quantize_bin), str(f16_path), str(gguf_path), quantization]
            logger.info("Quantizing to %s: %s", quantization, " ".join(cmd))
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Quantization failed: %s", result.stderr)
                sys.exit(1)

            # Clean up f16 intermediate
            f16_path.unlink(missing_ok=True)
        else:
            logger.warning(
                "llama-quantize not found, keeping f16 GGUF. "
                "Build llama.cpp or use --quantization f16."
            )
            f16_path.rename(gguf_path)
    else:
        # Method 2: Use mlx-lm fuse --export-gguf with base model + adapters
        # This requires the adapter path to exist alongside the base model
        adapter_path = Path(model_dir).parent / "adapters"
        base_model_cfg = None

        # Read the adapter config to find the base model
        adapter_config = adapter_path / "adapter_config.json"
        if adapter_path.exists() and adapter_config.exists():
            import json
            with open(adapter_config) as f:
                cfg = json.load(f)
            base_model_cfg = cfg.get("model", None)

        if adapter_path.exists():
            # Use mlx-lm fuse with adapters → direct GGUF export
            cmd = [
                sys.executable, "-m", "mlx_lm.fuse",
                "--model", base_model_cfg or model_dir,
                "--adapter-path", str(adapter_path),
                "--export-gguf",
                "--gguf-path", str(gguf_path),
            ]
            logger.info("Converting to GGUF with mlx-lm fuse: %s", " ".join(cmd))
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("mlx-lm fuse --export-gguf failed: %s", result.stderr)
                logger.info("stdout: %s", result.stdout)
                sys.exit(1)
        else:
            logger.error(
                "No conversion method available. Either:\n"
                "  1. Provide --llama-cpp-path pointing to llama.cpp checkout\n"
                "  2. Ensure adapter path exists at %s for mlx-lm fuse",
                adapter_path,
            )
            sys.exit(1)

    logger.info("GGUF file created: %s (%.1f MB)", gguf_path, gguf_path.stat().st_size / 1e6)
    return gguf_path


def create_modelfile(
    gguf_path: Path,
    output_dir: str,
    temperature: float = 0.1,
    num_ctx: int = 2048,
) -> Path:
    """Generate an Ollama Modelfile."""
    modelfile_path = Path(output_dir) / "Modelfile"

    # Qwen3 ChatML template. The model generates <think>...</think> as part
    # of its output (trained with empty think tags for non-reasoning tasks).
    # The template just provides the ChatML framing.
    template = (
        '{{- if .System }}<|im_start|>system\n'
        '{{ .System }}<|im_end|>\n'
        '{{ end }}'
        '{{- range .Messages }}'
        '{{- if eq .Role "user" }}<|im_start|>user\n'
        '{{ .Content }}<|im_end|>\n'
        '{{ else if eq .Role "assistant" }}<|im_start|>assistant\n'
        '{{ .Content }}<|im_end|>\n'
        '{{ end }}'
        '{{- end }}<|im_start|>assistant\n'
    )

    content = f"""FROM {gguf_path.resolve()}

TEMPLATE \"\"\"{template}\"\"\"

PARAMETER temperature {temperature}
PARAMETER top_p 0.95
PARAMETER num_ctx {num_ctx}
PARAMETER stop "<|im_end|>"
"""

    modelfile_path.write_text(content)
    logger.info("Modelfile written to %s", modelfile_path)
    return modelfile_path


def register_with_ollama(modelfile_path: Path, ollama_name: str) -> None:
    """Register the model with Ollama."""
    cmd = ["ollama", "create", ollama_name, "-f", str(modelfile_path)]
    logger.info("Registering with Ollama: %s", " ".join(cmd))

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ollama create failed: %s", result.stderr)
        sys.exit(1)

    logger.info("Model registered as '%s'", ollama_name)

    # Verify it's available
    result = subprocess.run(
        ["ollama", "list"],
        check=False, capture_output=True, text=True,
    )
    if ollama_name in result.stdout:
        logger.info("Verified: '%s' appears in ollama list", ollama_name)
    else:
        logger.warning("Model '%s' not found in ollama list output", ollama_name)


def run_smoke_test(ollama_name: str) -> None:
    """Run a quick smoke test on the registered model."""
    import litellm

    litellm.suppress_debug_info = True

    # Same test variables from compare_scoring_models.py
    test_prompt = """Score these variables:
1. [src/config.py] MAX_RETRIES = 3
2. [src/utils.py] _temp = []
3. [src/db.py] engine = create_engine(DATABASE_URL)
4. [src/app.py] logger = logging.getLogger(__name__)
5. [src/handlers.py] result = process_data(input_file)"""

    # Import training prompt (7-dimension) for testing
    training_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(training_dir))
    from generate_scoring_data import TEACHER_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": test_prompt},
    ]

    logger.info("Running smoke test with 5 test variables...")

    try:
        response = litellm.completion(
            model=f"ollama_chat/{ollama_name}",
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            api_base="http://localhost:11434",
            timeout=60,
            think=False,
        )

        content = response.choices[0].message.content or ""
        logger.info("Model output:\n%s", content)

        # Check format: expect 7 comma-separated scores per line
        score_pattern = re.compile(
            r"(\d+)\.\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)"
            r"\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)"
        )
        matches = list(score_pattern.finditer(content))
        logger.info("Parsed %d/5 score lines (7 dims each)", len(matches))

        if len(matches) >= 4:
            logger.info("Smoke test PASSED")
        else:
            logger.warning(
                "Smoke test WARNING: only %d/5 scores parsed. "
                "Model may need more training.",
                len(matches),
            )

        # Quick sanity checks
        for match in matches:
            idx = int(match.group(1))
            scores = tuple(int(match.group(i)) for i in range(2, 9))
            max_s = max(scores)

            if idx == 1 and max_s < 5:  # MAX_RETRIES should be high
                logger.warning("Var 1 (MAX_RETRIES) scored low: %s", scores)
            elif idx == 2 and max_s >= 5:  # _temp should be low
                logger.warning("Var 2 (_temp) scored high: %s", scores)
            elif idx == 5 and max_s >= 5:  # result = process_data should be low
                logger.warning("Var 5 (result) scored high: %s", scores)

    except Exception:
        logger.error("Smoke test FAILED", exc_info=True)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fine-tuned model to GGUF and register with Ollama."
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to merged HuggingFace model directory",
    )
    parser.add_argument(
        "--output-dir",
        default="tools/training/exports/",
        help="Output directory for GGUF and Modelfile",
    )
    parser.add_argument(
        "--quantization",
        default="Q4_K_M",
        help="GGUF quantization type (default: Q4_K_M)",
    )
    parser.add_argument(
        "--ollama-name",
        default="magaldi-scorer",
        help="Name to register in Ollama (default: magaldi-scorer)",
    )
    parser.add_argument(
        "--llama-cpp-path",
        help="Path to llama.cpp directory (for GGUF conversion)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run smoke test after registration",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Add project root to path for magaldi imports
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "src"))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Step 1: Convert to GGUF
    gguf_path = convert_to_gguf(
        args.model_dir,
        args.output_dir,
        args.quantization,
        args.llama_cpp_path,
    )

    # Step 2: Create Modelfile
    modelfile_path = create_modelfile(gguf_path, args.output_dir)

    # Step 3: Register with Ollama
    register_with_ollama(modelfile_path, args.ollama_name)

    # Step 4: Smoke test
    if args.test:
        run_smoke_test(args.ollama_name)

    logger.info("Export complete! Use model with: ollama run %s", args.ollama_name)


if __name__ == "__main__":
    main()
