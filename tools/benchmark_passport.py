#!/usr/bin/env python3
"""Standalone runner for passport embedding benchmark.

Preferred usage: magaldi-tools passport-benchmark [options]
Standalone:     PYTHONPATH=src python tools/benchmark_passport.py [options]
"""
from shared.cli.benchmark_passport import main

if __name__ == "__main__":
    main()
