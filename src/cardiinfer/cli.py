from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import InferAPI


def _load(path: str) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Request JSON must contain an object")
    return raw


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cardiinfer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report package and backend availability")
    sub.add_parser("backends", help="List discovered inference backends")
    infer = sub.add_parser("infer", help="Run an InferenceRequest JSON")
    infer.add_argument("request")
    propagate = sub.add_parser("propagate", help="Run an uncertainty-propagation request")
    propagate.add_argument("request")
    args = parser.parse_args(argv)
    api = InferAPI()

    if args.command == "doctor":
        _print(api.health())
        return 0
    if args.command == "backends":
        _print(api.backends())
        return 0
    if args.command == "infer":
        _print(api.infer(_load(args.request)))
        return 0
    if args.command == "propagate":
        _print(api.propagate(_load(args.request)))
        return 0
    return 2
