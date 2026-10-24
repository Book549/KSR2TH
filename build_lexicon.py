#!/usr/bin/env python3
# KSR2TH - S'gaw Karen Translator Using a Rule-Based
# Copyright (C) 2026 Wattanachai Maijampa
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the rule-based translator's lexicon from the dictionary workbook.

Output: rules/lexicon.csv  (thai, sgaw_karen, pos, confidence, source, category)

This is deliberately MORE permissive than the companion neural project's
`build_dict_csv.py`, which feeds that pipeline. There, a messy entry becomes a bad training example that
silently degrades the model. Here, every entry is a lookup the user can see
and audit, each row carries its own confidence rating, and multi-word Karen
values are perfectly usable - so the aim is coverage, with provenance
attached, rather than a small pristine set.

Rows are keyed by the THAI side, since Thai is the source language. Where a
Thai headword has several senses the alternates are all kept, ordered by the
workbook's own confidence rating (H > M > L); the translator uses the first
and reports the rest as alternates.
"""
from __future__ import annotations

import csv
import re
import sys
import unicodedata
from collections import defaultdict

import openpyxl

SRC = sys.argv[1] if len(sys.argv) > 1 else 'Sgaw_Karen_English_Thai_Dictionary2.xlsx'
OUT = 'rules/lexicon.csv'

MYA = re.compile(r'[က-႟]')
THAI = re.compile(r'[฀-๿]')
LATIN_OR_BRACKET = re.compile(r'[A-Za-z\[\]]')
POS_RE = re.compile(r'\bPOS:\s*\(?([a-z]+)\)?', re.I)

CATEGORY_POS = {
    'Locations': 'place', 'Animals': 'n', 'Food & drink': 'n',
    'Household items': 'n', 'Clothing': 'n', 'Transportation': 'n',
    'Body & health': 'n', 'Family': 'n', 'School & occupations': 'n',
    'Weather & nature': 'n', 'Adjectives': 'adj', 'Feelings': 'adj',
    'Colors (phonemic, academic source)': 'adj', 'Verbs': 'v',
    'Common verbs & hobbies': 'v', 'Numbers': 'num', 'Pronouns': 'pron',
    'Question words': 'qw', 'Time expressions': 'time', 'Days of the week': 'time',
    'Months': 'time', 'Directions': 'n', 'Greetings & courtesy': 'phrase',
}
POS_MAP = {'n': 'n', 'v': 'v', 'adj': 'adj', 'a': 'adj', 'adv': 'adv',
           'prep': 'prep', 'conj': 'conj', 'pron': 'pron', 'det': 'det', 'int': 'int'}
CONF_ORDER = {'H': 0, 'M': 1, 'L': 2, '': 3, None: 3}


# Some Karen cells are a NUMBERED sense list written with Karen digits, e.g.
# "၁ ကွၢ်ဃု, ဃုထံၣ် ၂ ထံၣ်န့ၢ်က့ၤ" (sense 1 ..., sense 2 ...). Splitting only on
# commas leaves the digits embedded in the output, which produced the garbage
# "၁ လဲၤလီၤ ၂ လီၤစၢၤ" for ตก. Only the FIRST numbered sense is kept.
SENSE_NUMBER = re.compile(r'(?:^|\s)[၀-၉]\s*')


def clean_side(cell) -> list:
    """Split a workbook cell into its alternate forms, cleaned.

    Editorial parentheses are removed but their CONTENTS are kept: the
    workbook uses them to mark an optional or variable slot, e.g. the greeting
    written "...တ(ဂီေ)အံေ...", where the enclosed piece is a real
    part of the word rather than a note about it.
    """
    s = unicodedata.normalize('NFC', str(cell)).strip()
    if LATIN_OR_BRACKET.search(s):
        return []
    s = s.replace('(', '').replace(')', '')
    if SENSE_NUMBER.search(s):
        pieces = [p for p in SENSE_NUMBER.split(s) if p.strip()]
        s = pieces[0] if pieces else s
    parts = [p.strip() for p in re.split(r'[,/]', s)]
    # A leftover sense digit means the split did not fully separate them;
    # drop rather than emit a value with stray numerals in it.
    return [p for p in parts if p and len(p) <= 40 and not SENSE_NUMBER.search(p)]


def main() -> None:
    wb = openpyxl.load_workbook(SRC, read_only=True)
    ws = wb['Dictionary']

    entries = defaultdict(list)
    stats = {'rows': 0, 'no_thai': 0, 'no_karen': 0, 'unusable': 0, 'kept': 0}

    for r in ws.iter_rows(min_row=2, values_only=True):
        stats['rows'] += 1
        category, thai_cell, karen_cell = r[0], r[2], r[3]
        conf, source, notes = r[7], r[8], r[9]
        if not thai_cell or not str(thai_cell).strip():
            stats['no_thai'] += 1; continue
        if not karen_cell or not str(karen_cell).strip():
            stats['no_karen'] += 1; continue

        thais = [t for t in clean_side(thai_cell) if THAI.search(t) and not MYA.search(t)]
        karens = [k for k in clean_side(karen_cell) if MYA.search(k) and not THAI.search(k)]
        if not thais or not karens:
            stats['unusable'] += 1; continue

        pos = CATEGORY_POS.get(str(category).strip()) if category else None
        if pos is None:
            m = POS_RE.search(str(notes or ''))
            pos = POS_MAP.get(m.group(1).lower()) if m else ''

        for th in thais:
            for kr in karens:
                entries[th].append({
                    'thai': th, 'sgaw_karen': kr, 'pos': pos or '',
                    'confidence': str(conf or ''), 'source': str(source or ''),
                    'category': str(category or ''),
                })

    rows = []
    for th in sorted(entries):
        seen = set()
        senses = sorted(entries[th], key=lambda e: CONF_ORDER.get(e['confidence'], 3))
        for e in senses:
            if e['sgaw_karen'] in seen:
                continue
            seen.add(e['sgaw_karen'])
            rows.append(e)
    stats['kept'] = len(rows)

    with open(OUT, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['thai', 'sgaw_karen', 'pos', 'confidence', 'source', 'category'])
        w.writeheader(); w.writerows(rows)

    import collections
    print(f'{len(entries)} distinct Thai headwords, {len(rows)} sense rows -> {OUT}')
    print('rejections:', stats)
    print('POS coverage:', dict(collections.Counter(r['pos'] or '(none)' for r in rows).most_common()))
    print('multi-word Thai headwords (phrases):',
          sum(1 for th in entries if len(th) > 12))


if __name__ == '__main__':
    main()
