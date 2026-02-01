#!/usr/bin/env python3
"""Entry point for the KB4096D system."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="KB4096D: Modular AI Knowledge Base in native dimensional space"
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name or path (default: TinyLlama-1.1B)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model (default: float16)",
    )
    parser.add_argument(
        "--kb-dir",
        default="knowledge_bases",
        help="Directory for KB files (default: knowledge_bases)",
    )
    parser.add_argument(
        "--extraction-layer",
        type=int,
        default=None,
        help="Layer for hidden state extraction (default: auto 2/3 depth)",
    )
    parser.add_argument(
        "--auto-init",
        action="store_true",
        help="Automatically load model on startup",
    )

    args = parser.parse_args()

    from kb4096d.config import KBConfig
    from kb4096d.cli import KB4096DCLI

    config = KBConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        kb_dir=Path(args.kb_dir),
        extraction_layer=args.extraction_layer,
    )

    cli = KB4096DCLI(config)

    if args.auto_init:
        cli.do_init("")

    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
