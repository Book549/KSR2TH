# KSR2TH - S'gaw Karen Translator Using a Rule-Based
# Copyright (C) 2026 Wattanachai Maijampa
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rule-based S'gaw Karen -> Thai translator (the reverse direction).

This is NOT the forward translator run backwards. Three things genuinely
differ, and each needed its own solution:

1. SEGMENTATION. There is no pythainlp for S'gaw Karen - no segmenter, no
   tagger, no corpus. Karen text is written with spaces, but they mark phrase
   boundaries rather than reliable word boundaries, so whitespace alone is not
   enough. Segmentation here is dictionary-driven: try multi-chunk phrases
   first, then longest-match inside each chunk.

2. AMBIGUITY RUNS THE OTHER WAY. One Karen word maps back to several Thai
   words far more often than the reverse. See reverse_lexicon.py - candidates
   are ranked by Thai corpus frequency, and the alternates are reported.

3. THE GRAMMAR RULES INVERT UNEVENLY. Karen's negation CIRCUMFIX collapses to
   a single Thai word, so the clause-final ဘၣ် must be found and deleted -
   which means looking ahead to the end of the clause, not just at the current
   token. Karen's allative ဆူ disappears entirely (Thai "ไปโรงเรียน" takes no
   preposition), and Karen's one general classifier has to be expanded back
   into the correct Thai classifier for the specific noun, which is
   information Karen simply does not carry.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import grammar as G
from .core_lexicon import CORE
from .lexicon import Lexicon, Sense
from .reverse_lexicon import ReverseLexicon

# ---------------------------------------------------------------------------
# Reverse core lexicon: Karen -> the CANONICAL Thai word.
#
# Built by inverting the hand-verified CORE table. Several Thai words map to
# the same Karen (ผม/ฉัน/ดิฉัน all -> ယ), so the FIRST one listed in CORE wins
# and becomes the canonical output - otherwise the choice would be arbitrary.
# ---------------------------------------------------------------------------
REVERSE_CORE: Dict[str, Tuple[str, str, str]] = {}
for _thai, (_karen, _pos, _conf, _note) in CORE.items():
    if _karen not in REVERSE_CORE:
        REVERSE_CORE[_karen] = (_thai, _pos, _conf)

def _invert_first_wins(mapping: Dict[str, str]) -> Dict[str, str]:
    """Invert a Thai->Karen map, keeping the FIRST Thai spelling listed.

    Several Thai forms share one Karen word (ที่ไหน and ไหน are both ဖဲလဲၣ်).
    A plain dict comprehension keeps the LAST one, which produced the clipped
    "คุณไปไหน" instead of "คุณไปที่ไหน". First-wins makes the canonical form
    the one that comes back.
    """
    out: Dict[str, str] = {}
    for thai, karen in mapping.items():
        out.setdefault(karen, thai)
    return out


# Karen digits -> Thai number words (the forward direction's map, inverted).
KAREN_TO_THAI_NUMBER = _invert_first_wins(G.THAI_NUMBERS)
# Karen fused wh-word + question marker -> Thai wh-word.
KAREN_TO_THAI_QUESTION = _invert_first_wins(G.QUESTION_WORDS)

# Thai classifiers by noun. Karen carries only ONE general classifier, so the
# Thai classifier has to be reconstructed from the noun - information the
# Karen sentence does not contain. Known nouns get their real classifier;
# anything else falls back to อัน and is flagged, rather than silently
# emitting a classifier that may be wrong.
THAI_CLASSIFIER_BY_NOUN: Dict[str, str] = {
    "หมา": "ตัว", "สุนัข": "ตัว", "แมว": "ตัว", "ช้าง": "ตัว", "นก": "ตัว",
    "ปลา": "ตัว", "ไก่": "ตัว", "วัว": "ตัว", "ควาย": "ตัว", "ม้า": "ตัว",
    "รถยนต์": "คัน", "รถ": "คัน", "จักรยาน": "คัน", "ร่ม": "คัน",
    "หนังสือ": "เล่ม", "สมุด": "เล่ม", "ดินสอ": "แท่ง",
    "ฟุตบอล": "ลูก", "ลูกบอล": "ลูก", "ไข่": "ฟอง",
    "กาแฟ": "แก้ว", "ชา": "แก้ว", "น้ำ": "แก้ว",
    "คอมพิวเตอร์": "เครื่อง", "โทรศัพท์": "เครื่อง",
    "ภาพยนตร์": "เรื่อง", "หนัง": "เรื่อง", "เพลง": "เพลง", "ดนตรี": "เพลง",
    "บ้าน": "หลัง", "ครู": "คน", "นักเรียน": "คน", "คน": "คน",
    "ต้นไม้": "ต้น", "ดอกไม้": "ดอก", "ผลไม้": "ผล",
}
DEFAULT_CLASSIFIER = "อัน"


@dataclass
class RToken:
    karen: str
    thai: str = ""
    role: str = ""
    sense: Optional[Sense] = None
    alternates: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def known(self) -> bool:
        return self.role != "unknown"


@dataclass
class RTranslation:
    source: str
    thai: str
    tokens: List[RToken] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def unknown_words(self) -> List[str]:
        return [t.karen for t in self.tokens if t.role == "unknown"]

    @property
    def coverage(self) -> float:
        content = [t for t in self.tokens if t.role != "dropped"]
        if not content:
            return 1.0
        return sum(1 for t in content if t.known) / len(content)

    @property
    def ambiguous(self) -> List[Tuple[str, List[str]]]:
        """Tokens where a choice was made between several Thai words."""
        return [(t.karen, t.alternates) for t in self.tokens if t.alternates]


class KarenToThai:
    NAMES_PATH = os.path.join(os.path.dirname(__file__), "names.txt")

    def __init__(self, lexicon: Optional[Lexicon] = None, names_path: Optional[str] = None):
        self.lex = lexicon or Lexicon()
        self.rev = ReverseLexicon(self.lex)
        self.names = self._load_names(names_path or self.NAMES_PATH)
        # Every Karen string the segmenter is allowed to recognise.
        self.vocab = set(self.rev.headwords) | set(REVERSE_CORE) | self.names
        self.vocab |= set(KAREN_TO_THAI_NUMBER) | set(KAREN_TO_THAI_QUESTION)
        self.vocab |= {G.NEG_PRE, G.NEG_POST, G.FUTURE, G.PERFECT, G.INTENSIFIER,
                       G.CLASSIFIER, G.DEMONSTRATIVE, G.LOCATIVE, G.ALLATIVE, G.GO}
        self._longest = max((len(w) for w in self.vocab), default=1)

    @staticmethod
    def _load_names(path: str) -> set:
        names = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.split("#", 1)[0].strip()
                    if line:
                        names.add(line)
        return names

    # ------------------------------------------------------------------
    def segment(self, text: str) -> List[str]:
        """Split Karen text into known units.

        Karen writes spaces, but they are phrase boundaries as much as word
        boundaries, and a multi-word dictionary entry can span them. So
        whitespace chunks are first tried in groups (longest group first),
        and only then is a single chunk broken up internally.
        """
        chunks = text.split()
        tokens: List[str] = []
        i = 0
        while i < len(chunks):
            for size in (3, 2):
                if i + size <= len(chunks):
                    joined = " ".join(chunks[i:i + size])
                    if joined in self.vocab:
                        tokens.append(joined)
                        i += size
                        break
            else:
                chunk = chunks[i]
                if chunk in self.vocab:
                    tokens.append(chunk)
                else:
                    tokens.extend(self._split_chunk(chunk))
                i += 1
                continue
        return tokens

    def _split_chunk(self, chunk: str) -> List[str]:
        """Longest-match inside one whitespace chunk.

        Karen text is not always spaced word by word, so a chunk can hold
        several words. Anything that cannot be matched is kept as one unknown
        piece rather than shredded into fragments that would each be reported
        as a separate missing word.
        """
        out: List[str] = []
        i, pending = 0, ""
        while i < len(chunk):
            for size in range(min(self._longest, len(chunk) - i), 0, -1):
                piece = chunk[i:i + size]
                if piece in self.vocab and size > 1:
                    if pending:
                        out.append(pending); pending = ""
                    out.append(piece)
                    i += size
                    break
            else:
                pending += chunk[i]
                i += 1
        if pending:
            out.append(pending)
        return out

    # ------------------------------------------------------------------
    def classify(self, tokens: List[str]) -> List[RToken]:
        out: List[RToken] = []
        for word in tokens:
            t = RToken(karen=word)
            if word in self.names:
                t.role, t.thai = "name", word
                t.note = "proper name, copied through untranslated"
            elif word == G.NEG_PRE:
                t.role, t.thai = "grammar", "NEG"
            elif word == G.NEG_POST:
                t.role, t.thai = "grammar", "NEG_END"
            elif word == G.FUTURE:
                t.role, t.thai = "grammar", "FUT"
            elif word == G.PERFECT:
                t.role, t.thai = "grammar", "PERF"
            elif word == G.INTENSIFIER:
                t.role, t.thai = "grammar", "มาก"
            elif word == G.DEMONSTRATIVE:
                t.role, t.thai = "grammar", "นี้"
            elif word == G.LOCATIVE:
                t.role, t.thai = "grammar", "ที่"
            elif word == G.ALLATIVE:
                t.role, t.thai = "grammar", "ALLATIVE"
            elif word == G.CLASSIFIER:
                t.role, t.thai = "classifier", ""
            elif word in KAREN_TO_THAI_QUESTION:
                t.role, t.thai = "question", KAREN_TO_THAI_QUESTION[word]
            elif word in KAREN_TO_THAI_NUMBER:
                t.role, t.thai = "number", KAREN_TO_THAI_NUMBER[word]
            elif word in REVERSE_CORE:
                thai, pos, conf = REVERSE_CORE[word]
                t.role, t.thai = "core", thai
                t.sense = Sense(thai=thai, karen=word, pos=pos, confidence=conf,
                                source="hand-verified core lexicon", category="")
            else:
                sense = self.rev.lookup(word)
                if sense is not None:
                    t.role, t.thai, t.sense = "lexicon", sense.thai, sense
                    others = [s.thai for s in self.rev.candidates(word)[1:4]]
                    if others:
                        t.alternates = others
                else:
                    t.role = "unknown"
                    t.note = "not in the dictionary; passed through as Karen"
                    t.thai = word
            out.append(t)
        return out

    # ------------------------------------------------------------------
    def assemble(self, tokens: List[RToken]) -> Tuple[str, List[str]]:
        out: List[str] = []
        notes: List[str] = []
        # The clause-final negation partner has no Thai counterpart at all -
        # Thai marks negation once, before the verb. Find it up front so it
        # can be deleted rather than translated.
        has_neg_end = any(t.thai == "NEG_END" for t in tokens)
        pending_neg = False
        last_noun: Optional[str] = None

        for i, t in enumerate(tokens):
            if t.role == "grammar":
                if t.thai == "NEG":
                    pending_neg = True
                    continue
                if t.thai == "NEG_END":
                    continue  # collapses into the single Thai ไม่
                if t.thai == "FUT":
                    out.append("จะ")
                    continue
                if t.thai == "PERF":
                    out.append("แล้ว")
                    continue
                if t.thai == "ALLATIVE":
                    # Thai takes no preposition here: "ไปโรงเรียน", not
                    # "ไปที่โรงเรียน". Dropped rather than translated.
                    continue
                out.append(t.thai)
                continue

            if t.role == "classifier":
                clf = THAI_CLASSIFIER_BY_NOUN.get(last_noun or "")
                if clf is None:
                    clf = DEFAULT_CLASSIFIER
                    notes.append(
                        f"Karen uses one general classifier, so the Thai classifier for "
                        f"'{last_noun or '?'}' could not be recovered; used the generic '{clf}'"
                    )
                out.append(clf)
                t.thai = clf   # so --explain shows what was actually emitted
                continue

            if pending_neg:
                out.append("ไม่")
                pending_neg = False

            out.append(t.thai)
            if t.role in ("lexicon", "core", "name") and t.thai:
                last_noun = t.thai

        if pending_neg:
            out.append("ไม่")
        if has_neg_end and "ไม่" not in out:
            notes.append("a clause-final ဘၣ် was present without its partner တ; "
                         "the sentence may not have been negated as intended")

        # Thai is written without spaces between words.
        return "".join(out), notes

    # ------------------------------------------------------------------
    def translate(self, text: str) -> RTranslation:
        text = " ".join(text.split())
        if not text:
            return RTranslation(source=text, thai="")
        tokens = self.classify(self.segment(text))
        thai, notes = self.assemble(tokens)

        unknown = [t.karen for t in tokens if t.role == "unknown"]
        if unknown:
            notes.append("not in the dictionary, passed through as Karen: " + ", ".join(unknown))
        amb = [(t.karen, t.thai, t.alternates) for t in tokens if t.alternates]
        for karen, chosen, others in amb[:6]:
            notes.append(f"'{karen}' could also be {', '.join(others)} - chose '{chosen}' "
                         f"as the commonest Thai word")
        return RTranslation(source=text, thai=thai, tokens=tokens, notes=notes)
