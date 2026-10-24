# KSR2TH - S'gaw Karen Translator Using a Rule-Based
# Copyright (C) 2026 Wattanachai Maijampa
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thai -> S'gaw Karen lexicon lookup, with provenance kept on every entry."""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "lexicon.csv")


@dataclass(frozen=True)
class Sense:
    """One Thai -> Karen sense, carrying the workbook's own rating."""
    thai: str
    karen: str
    pos: str
    confidence: str   # H / M / L, straight from the workbook
    source: str       # e.g. "PICDICT95;BFAS-APK;MOBI"
    category: str

    def __str__(self) -> str:
        return self.karen


class Lexicon:
    """All Thai headwords, each with its ordered list of senses."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self.entries: Dict[str, List[Sense]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Lexicon not found at {self.path}. Build it with:\n"
                f"    python build_lexicon.py Sgaw_Karen_English_Thai_Dictionary2.xlsx"
            )
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                self.entries[row["thai"]].append(Sense(
                    thai=row["thai"], karen=row["sgaw_karen"], pos=row.get("pos", ""),
                    confidence=row.get("confidence", ""), source=row.get("source", ""),
                    category=row.get("category", ""),
                ))

    # ------------------------------------------------------------------
    def __contains__(self, thai: str) -> bool:
        return thai in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def headwords(self) -> List[str]:
        return list(self.entries)

    def senses(self, thai: str) -> List[Sense]:
        return list(self.entries.get(thai, []))

    def lookup(self, thai: str, pos: str = "") -> Optional[Sense]:
        """Best sense for a Thai word, preferring one whose part of speech
        matches what the tagger said.

        Sense selection is the main place a word-for-word translator goes
        wrong: a Thai word that is both a noun and a verb has different Karen
        equivalents, and picking the wrong one silently changes the meaning.
        Using the tagger's POS narrows that down. Within a POS, senses are
        already ordered by the workbook's confidence rating (H before M
        before L), so the best-attested reading wins.
        """
        senses = self.entries.get(thai)
        if not senses:
            return None
        if pos:
            for s in senses:
                if s.pos == pos:
                    return s
            # An adjective in Karen IS a stative verb, so a verb entry is a
            # reasonable stand-in for an adjective reading and vice versa.
            if pos in ("adj", "v"):
                for s in senses:
                    if s.pos in ("adj", "v"):
                        return s
        return senses[0]
