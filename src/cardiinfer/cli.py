from __future__ import annotations

import argparse
import json

from .api import InferAPI


def main() -> int:
    parser = argparse.ArgumentParser(prog="cardiinfer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report package and backend availability")
    args = parser.parse_args()

    if args.command == "doctor":
        print(json.dumps(InferAPI().health(), indent=2, sort_keys=True))
        return 0
    return 2
