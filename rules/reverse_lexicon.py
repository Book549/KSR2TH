# KSR2TH - S'gaw Karen Translator Using a Rule-Based
# Copyright (C) 2026 Wattanachai Maijampa
# SPDX-License-Identifier: AGPL-3.0-or-later
"""S'gaw Karen -> Thai lookup: the lexicon inverted, with ambiguity ranked.

INVERTING A DICTIONARY IS NOT SYMMETRIC
---------------------------------------
Going Thai -> Karen, a Thai headword's senses are already ordered by the
workbook's own confidence rating and the first is usually right. Going the
other way, one Karen word routinely maps back to SEVERAL Thai words - 513 of
the 6,259 Karen headwords (8.2%) - and the workbook offers no ordering among
them at all, because they came from different rows.

Confidence alone cannot break those ties: the candidates are often all rated
the same. So candidates are ranked by how COMMON the Thai word actually is,
using the Thai National Corpus frequency list bundled offline with pythainlp -
the same corpus behind this project's Thai word frequency ranking.
For "dog" that picks หมา (3,752 occurrences) over สุนัข (1,109), which is the
word a reader would expect.

This is a heuristic, not a disambiguator: it has no idea what the sentence is
about. Where it matters, the alternates are reported so a reader can see that
a choice was made.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from .lexicon import Lexicon, Sense

CONF_RANK = {"H": 0, "M": 1, "L": 2, "": 3}

_TNC: Optional[Dict[str, int]] = None


def thai_frequency() -> Dict[str, int]:
    """Thai National Corpus counts, loaded once. Empty dict if unavailable."""
    global _TNC
    if _TNC is None:
        try:
            from pythainlp.corpus.tnc import word_freqs
            _TNC = dict(word_freqs())
        except Exception:
            _TNC = {}
    return _TNC


class ReverseLexicon:
    """Karen headword -> Thai candidates, best first."""

    def __init__(self, lexicon: Optional[Lexicon] = None):
        self.lex = lexicon or Lexicon()
        freq = thai_frequency()
        buckets: Dict[str, List[Sense]] = defaultdict(list)
        for senses in self.lex.entries.values():
            for s in senses:
                buckets[s.karen].append(s)

        self.entries: Dict[str, List[Sense]] = {}
        for karen, senses in buckets.items():
            seen, unique = set(), []
            for s in senses:
                if s.thai in seen:
                    continue
                seen.add(s.thai)
                unique.append(s)
            # Ranking order matters and was chosen by testing, not assumed.
            # Sorting by confidence FIRST looked principled but picked สถาบัน
            # ("institution", H) over โรงเรียน ("school", M) for ကၠိ - the
            # wrong word for almost every sentence. H and M are both
            # "attested", and between two attested Thai synonyms the useful
            # signal is which one Thai speakers actually use. So only L is
            # demoted, and corpus frequency decides among the rest.
            unique.sort(key=lambda s: (
                1 if s.confidence == "L" else 0,  # demote low-confidence rows
                -freq.get(s.thai, 0),             # prefer the commoner Thai word
                len(s.thai),                      # then the simpler one
            ))
            self.entries[karen] = unique

    def __contains__(self, karen: str) -> bool:
        return karen in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def headwords(self) -> List[str]:
        return list(self.entries)

    def candidates(self, karen: str) -> List[Sense]:
        return list(self.entries.get(karen, []))

    def lookup(self, karen: str, pos: str = "") -> Optional[Sense]:
        cands = self.entries.get(karen)
        if not cands:
            return None
        if pos:
            for s in cands:
                if s.pos == pos:
                    return s
        return cands[0]
