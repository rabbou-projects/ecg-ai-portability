#!/usr/bin/env python3
"""Count the personal literals and machine paths written into this repo's tracked files.

A value of one person or one machine written into a repository breaks on any
other checkout and leaks wherever the repository goes. This ratchet counts four
kinds over every tracked text file that is not prose (Markdown is prose):

``home_path``
    An absolute path into a user's home (``/Users/<name>``, ``/home/<name>``)
    or a tailnet address (``100.64.0.0/10``): it resolves on one machine only.
``contact``
    The owner's e-mail address, phone number or birth date, matched by hash so
    this file never holds them; any IBAN.
``name``
    The owner's first name or surname as a word, matched by hash.

The count per kind and per file lives in ``hardcode-baseline.json`` next to the
repo root. ``--check`` fails when a file holds more than its baseline, when a
file not in the baseline holds any, or when a file holds fewer (the baseline is
lowered in the same change, so the count can only fall). ``--lower`` rewrites
the baseline to the current counts and refuses if any count would rise.

Standard library only, so every repo's CI can run it with ``python3``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASELINE_NAME = "hardcode-baseline.json"
KINDS = ("home_path", "contact", "name")

# sha256 prefixes (16 hex) of the normalised literal, so no literal is written here.
EMAIL_HASHES = frozenset({"dfed95680298eb3e"})
PHONE_HASHES = frozenset({"e5fe76b9afe31cc0"})  # national form, digits only
DATE_HASHES = frozenset({"5c9a7f8609fb3cda"})  # ISO date
NAME_HASHES = frozenset({"b04e95d0b09d1c38", "711ea5f3db989de2"})  # lowercase word

HOME_PATH = re.compile(r"(?<![\w.~$-])/(?:Users|home)/(?!runner\b)[A-Za-z][\w.-]*")
TAILNET = re.compile(r"(?<![\d.])100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}(?![\d.])")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<![\d])(?:\+33[\s.-]?|0)[1-9](?:[\s.-]?\d{2}){4}(?![\d])")
ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
FR_DATE = re.compile(r"(?<!\d)(\d{2})/(\d{2})/(\d{4})(?!\d)")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
WORD = re.compile(r"(?<![\w])[A-Za-zÀ-ÿ]{5}(?![\w])")

PROSE_SUFFIXES = (".md", ".mdx", ".markdown", ".rst")
SELF_FILES = frozenset({"scripts/hardcode_ratchet.py", BASELINE_NAME})
MAX_BYTES = 2_000_000


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _iban_valid(raw: str) -> bool:
    s = raw.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    digits = "".join(str(int(c, 36)) for c in rearranged)
    return int(digits) % 97 == 1


def count_text(
    text: str,
    *,
    emails: frozenset[str] = EMAIL_HASHES,
    phones: frozenset[str] = PHONE_HASHES,
    dates: frozenset[str] = DATE_HASHES,
    names: frozenset[str] = NAME_HASHES,
) -> Counter[str]:
    """Count each kind's matches in one file's text."""
    out: Counter[str] = Counter()
    out["home_path"] += len(HOME_PATH.findall(text)) + len(TAILNET.findall(text))
    contact = sum(_h(m.lower()) in emails for m in EMAIL.findall(text))
    for m in PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        national = "0" + digits[2:] if digits.startswith("33") else digits
        contact += _h(national) in phones
    contact += sum(_h(f"{y}-{mo}-{d}") in dates for y, mo, d in ISO_DATE.findall(text))
    contact += sum(_h(f"{y}-{mo}-{d}") in dates for d, mo, y in FR_DATE.findall(text))
    contact += sum(_iban_valid(m) for m in IBAN.findall(text))
    out["contact"] += contact
    out["name"] += sum(_h(w.lower()) in names for w in WORD.findall(text))
    return +out


def git_env() -> dict[str, str]:
    """The environment without the GIT_* variables a git hook exports, which aim git at the hooked repo."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def tracked_files(root: Path) -> list[str]:
    res = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True, env=git_env()
    )
    return [p for p in res.stdout.decode().split("\0") if p]


def scan(root: Path, *, include_prose: bool = False) -> dict[str, dict[str, int]]:
    """Per kind, per tracked file, the count of matches (files with zero are absent)."""
    out: dict[str, dict[str, int]] = {k: {} for k in KINDS}
    for rel in tracked_files(root):
        if rel in SELF_FILES or rel.endswith("test_hardcode_ratchet.py"):
            continue
        if not include_prose and rel.lower().endswith(PROSE_SUFFIXES):
            continue
        path = root / rel
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_BYTES:
            continue
        raw = path.read_bytes()
        if b"\0" in raw[:8192]:
            continue
        for kind, n in count_text(raw.decode("utf-8", "replace")).items():
            out[kind][rel] = n
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def compare(current: dict[str, dict[str, int]], baseline: dict[str, dict[str, int]]) -> list[str]:
    """Every (kind, file) whose count differs from the baseline, as a message."""
    errors: list[str] = []
    for kind in KINDS:
        cur, base = current.get(kind, {}), baseline.get(kind, {})
        for rel in sorted(set(cur) | set(base)):
            c, b = cur.get(rel, 0), base.get(rel, 0)
            if c > b:
                errors.append(f"RISE {kind} {rel}: {c} > baseline {b}; move the value to config, the store or the secret store")
            elif c < b:
                errors.append(f"FALL {kind} {rel}: {c} < baseline {b}; run --lower and commit {BASELINE_NAME}")
    return errors


def totals(counts: dict[str, dict[str, int]]) -> dict[str, int]:
    return {k: sum(counts.get(k, {}).values()) for k in KINDS}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail on any difference from the baseline")
    mode.add_argument("--lower", action="store_true", help="write the current counts; refuse a rise")
    mode.add_argument("--report", action="store_true", help="print totals, prose included, as JSON")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    if args.report:
        print(json.dumps({"code": totals(scan(root)), "prose": totals(scan(root, include_prose=True))}))
        return 0
    current = scan(root)
    bpath = root / BASELINE_NAME
    baseline = json.loads(bpath.read_text()) if bpath.exists() else {}
    if args.lower:
        rises = [e for e in compare(current, baseline) if e.startswith("RISE")] if baseline else []
        if rises:
            print("\n".join(rises), file=sys.stderr)
            return 1
        bpath.write_text(json.dumps(current, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps(totals(current)))
        return 0
    if not bpath.exists():
        print(f"no {BASELINE_NAME}; run --lower once to create it", file=sys.stderr)
        return 2
    errors = compare(current, baseline)
    print(json.dumps({"baseline": totals(baseline), "current": totals(current)}))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
