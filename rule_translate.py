#!/usr/bin/env python3
# KSR2TH - S'gaw Karen Translator Using a Rule-Based
# Copyright (C) 2026 Wattanachai Maijampa
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rule-based translator, both directions (no model, no training).

    python rule_translate.py --text "ผมไม่ชอบกาแฟ"                       # Thai -> Karen
    python rule_translate.py --direction kr2th --text "ယ တ ဘၣ်သး ခီဖံၣ် ဘၣ်"  # Karen -> Thai
    python rule_translate.py --text "..." --explain
    python rule_translate.py --interactive
    python rule_translate.py --file sentences.txt --json out.json

Every output word comes from a dictionary row or a grammar rule, and --explain
shows which, for every token. Words that are not in the dictionary are passed
through as Thai and listed, so a gap is always visible rather than silently
filled with a wrong guess.
"""
from __future__ import annotations

import argparse
import json
import sys

from rules.reverse import KarenToThai
from rules.translator import RuleTranslator

ROLE_LABEL = {
    "core": "core lexicon (hand-verified)",
    "lexicon": "dictionary",
    "grammar": "grammar rule",
    "question": "question word + clause-final marker",
    "number": "numeral -> Karen digit",
    "classifier": "classifier",
    "name": "proper name (copied)",
    "dropped": "dropped",
    "unknown": "NOT IN DICTIONARY",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rule-based Thai -> S'gaw Karen translation.")
    p.add_argument("--direction", type=str, default="th2kr", choices=["th2kr", "kr2th"],
                   help="th2kr = Thai -> S'gaw Karen (default); kr2th = S'gaw Karen -> Thai.")
    p.add_argument("--text", type=str, default=None, help="A single sentence to translate.")
    p.add_argument("--file", type=str, default=None, help="Translate each line of a file.")
    p.add_argument("--interactive", action="store_true", help="Interactive prompt.")
    p.add_argument("--explain", action="store_true", help="Show the per-token decisions.")
    p.add_argument("--json", type=str, default=None, help="Write results to a JSON file.")
    return p.parse_args()


def show(translator, text: str, explain: bool, direction: str) -> dict:
    r = translator.translate(text)
    forward = direction == "th2kr"
    src_label, tgt_label = ("TH", "KR") if forward else ("KR", "TH")
    target = r.karen if forward else r.thai
    print(f"\n{src_label}  {r.source}")
    print(f"{tgt_label}  {target}")
    extra = f"   lowest source confidence: {r.lowest_confidence}" if forward else ""
    print(f"    coverage {r.coverage:.0%}{extra}")
    if explain:
        print("    " + "-" * 66)
        for t in r.tokens:
            label = ROLE_LABEL.get(t.role, t.role)
            conf = f" [{t.sense.confidence}]" if t.sense else ""
            a, b = (t.thai, t.karen) if forward else (t.karen, t.thai)
            # An unknown word is copied through, not dropped - showing
            # "(dropped)" for it contradicted the note printed underneath.
            emitted = b or (a if t.role == "unknown" else "(dropped)")
            print(f"    {a:<14} -> {emitted:<22} {label}{conf}")
            if getattr(t, "alternates", None):
                print(f"    {'':<14}    also possible: {', '.join(t.alternates)}")
            if t.note:
                print(f"    {'':<14}    {t.note}")
    for n in r.notes:
        print(f"    ! {n}")
    out = {"source": r.source, "translation": target,
           "coverage": round(r.coverage, 4),
           "unknown_words": r.unknown_words, "notes": r.notes}
    if forward:
        out["lowest_confidence"] = r.lowest_confidence
    return out


def main() -> None:
    args = parse_args()
    forward = args.direction == "th2kr"
    translator = RuleTranslator() if forward else KarenToThai()
    size = len(translator.lex) if forward else len(translator.rev)
    unit = "Thai headwords" if forward else "S'gaw Karen headwords"
    print(f"{'Thai -> Karen' if forward else 'Karen -> Thai'} · "
          f"{size} {unit} · {len(translator.names)} proper names", file=sys.stderr)

    results = []
    if args.interactive:
        print("Type Thai text, or 'quit' to exit.")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if text.lower() in {"quit", "exit"}:
                break
            if text:
                results.append(show(translator, text, args.explain, args.direction))
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(show(translator, line, args.explain, args.direction))
    elif args.text:
        results.append(show(translator, args.text, args.explain, args.direction))
    else:
        raise SystemExit('Provide --text "...", --file PATH, or --interactive')

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {len(results)} results to {args.json}")


if __name__ == "__main__":
    main()
