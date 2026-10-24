# KSR2TH - S'gaw Karen Translator Using a Rule-Based
# Copyright (C) 2026 Wattanachai Maijampa
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rule-based Thai -> S'gaw Karen translator.

Pipeline: segment -> tag -> look up -> apply transfer rules -> assemble.

There is no model here and nothing is learned. Every output word came from a
dictionary row or from a grammar rule, and the translator can say which for
every token it produced. That is the point: on a low-resource pair with a good
dictionary and no real parallel corpus, an auditable rule engine beats a model
trained on synthetic sentences, and it improves the moment a dictionary row is
added rather than requiring a retrain.

WHAT IT CANNOT DO, stated plainly:
  * It translates word by word with local rules. It has no model of meaning,
    so it will not resolve an ambiguous word from context beyond its
    part of speech, and it will not reorder anything the rules do not cover.
  * Long or syntactically complex sentences (relative clauses, coordination,
    embedded speech) are passed through largely word-for-word. The result is
    usually understandable but not fluent.
  * Coverage is bounded by the dictionary: 3,446 Thai headwords.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import grammar as G
from .core_lexicon import ALREADY_NEGATIVE, CORE
from .lexicon import Lexicon, Sense

try:
    from pythainlp.tag import pos_tag
    from pythainlp.tokenize import Tokenizer, word_tokenize
    from pythainlp.util import dict_trie
    _PYTHAINLP = True
except ImportError:  # pragma: no cover
    _PYTHAINLP = False


@dataclass
class Token:
    """One source token and what the translator decided to do with it."""
    thai: str
    tag: str = ""            # pythainlp ORCHID tag
    pos: str = ""            # our POS code
    karen: str = ""          # what was emitted (empty = dropped)
    role: str = ""           # lexicon | grammar | number | classifier | question | name | unknown | dropped
    sense: Optional[Sense] = None
    note: str = ""

    @property
    def known(self) -> bool:
        return self.role not in ("unknown",)


@dataclass
class Translation:
    source: str
    karen: str
    tokens: List[Token] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def unknown_words(self) -> List[str]:
        return [t.thai for t in self.tokens if t.role == "unknown"]

    @property
    def coverage(self) -> float:
        """Share of content tokens that were actually translated."""
        content = [t for t in self.tokens if t.role != "dropped"]
        if not content:
            return 1.0
        return sum(1 for t in content if t.known) / len(content)

    @property
    def lowest_confidence(self) -> str:
        ranks = {"H": 0, "M": 1, "L": 2}
        worst, label = -1, "-"
        for t in self.tokens:
            if t.sense and t.sense.confidence in ranks:
                if ranks[t.sense.confidence] > worst:
                    worst, label = ranks[t.sense.confidence], t.sense.confidence
        return label


class RuleTranslator:
    NAMES_PATH = os.path.join(os.path.dirname(__file__), "names.txt")

    def __init__(self, lexicon: Optional[Lexicon] = None, names_path: Optional[str] = None):
        self.lex = lexicon or Lexicon()
        self.names = self._load_names(names_path or self.NAMES_PATH)
        self._words = self._segmenter_words()
        self._tokenizer = None
        if _PYTHAINLP:
            from pythainlp.corpus.common import thai_words
            self._tokenizer = Tokenizer(
                dict_trie(self._drop_glued_negatives(set(thai_words())) | self._words),
                engine="newmm")

    def _segmenter_words(self) -> set:
        """Every form the segmenter may match, used by BOTH segmentation paths.

        The pythainlp tokenizer is seeded with these so multi-word dictionary
        entries ("สวัสดีตอนเช้า") survive as single units instead of being
        split into pieces we would then fail to find. The fallback matcher
        tests against the same set, so the two paths agree on what a word is.
        Function words are included because the grammar rules - not the
        dictionary - are what consume them: without "จะ" in here the fallback
        matcher did not recognise it, and "เขาจะไปโรงเรียน" came out with an
        unknown "จะไป" in place of the future marker and the verb.
        """
        words = set(self.lex.headwords) | set(CORE) | self.names
        words |= G.NEGATION_WORDS | G.FUTURE_WORDS | G.PERFECT_WORDS
        words |= G.INTENSIFIER_WORDS | G.DEMONSTRATIVE_WORDS | G.LOCATIVE_WORDS
        words |= G.GO_WORDS | G.POLITENESS_WORDS | G.POLAR_QUESTION_WORDS
        words |= G.UNSOURCED_QUESTION_WORDS | G.THAI_CLASSIFIERS
        words |= set(G.QUESTION_WORDS) | set(G.THAI_NUMBERS)
        return self._drop_glued_negatives(words)

    @staticmethod
    def _drop_glued_negatives(words: set) -> set:
        """Remove headwords that glue a negation or future particle to a verb.

        The workbook contains entries like "ไม่ชอบ" (dislike). If the
        segmenter is allowed to keep those whole, negation gets resolved
        LEXICALLY and the Karen circumfix is never built - the sentence comes
        out missing its clause-final ဘၣ်. Found by testing "ผมไม่ชอบกาแฟ",
        which produced no closing particle at all. Dropping them forces
        negation through the grammar rules, where it is handled once and
        consistently.
        """
        return {w for w in words
                if not any(w.startswith(p) and len(w) > len(p)
                           for p in ("ไม่", "จะ"))}

    @staticmethod
    def _load_names(path: str) -> set:
        """Proper names to copy through untouched."""
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
        if self._tokenizer is not None:
            toks = self._tokenizer.word_tokenize(text)
        elif _PYTHAINLP:
            toks = word_tokenize(text, engine="newmm")
        else:
            toks = self._maximal_match(text)
        return [t for t in (x.strip() for x in toks) if t]

    # Inside a stretch no dictionary entry covers, a hit shorter than this is
    # ignored. Thai is written without spaces, so a one- or two-character
    # entry matches somewhere inside almost any unknown word: "สวัสดี"
    # (hello) is not a headword but it ends in "ดี" (good), and matching that
    # cut the greeting into "สวัส" + "good". Content words are longer than
    # two characters, so requiring three keeps the unknown word whole.
    MIN_INTERNAL_MATCH = 3

    # An unknown stretch shorter than this is not a word either, so a
    # dictionary hit is not allowed to cut one off. "อย่างไรบ้าง" (how is it?)
    # contains "ย่าง" from the second character on, which left a bare "อ"
    # standing alone as its own unknown token.
    MIN_UNKNOWN_CHUNK = 2

    @staticmethod
    def _is_thai(ch: str) -> bool:
        return "\u0e00" <= ch <= "\u0e7f"

    def _longest_at(self, text: str, i: int, longest: int) -> str:
        """The longest dictionary entry starting at i, or "" if there is none."""
        for size in range(min(longest, len(text) - i), 1, -1):
            chunk = text[i:i + size]
            if self._known(chunk):
                return chunk
        return text[i] if self._known(text[i]) else ""

    def _maximal_match(self, text: str) -> List[str]:
        """Longest-match fallback if pythainlp is unavailable.

        Characters that no dictionary entry covers are kept together as ONE
        token instead of being emitted one at a time. Emitting them singly is
        what turned an unknown "สวัสดี" into the six tokens
        "ส ว ั ส ด ี" - which is also how it came out of the assembler. A
        lone vowel mark is not a word in any script, so a split that produces
        one is always wrong.
        """
        words: List[str] = []
        longest = max((len(w) for w in self.lex.headwords), default=1)
        for run in text.split():
            i = 0
            while i < len(run):
                match = self._longest_at(run, i, longest)
                if match:
                    words.append(match)
                    i += len(match)
                    continue
                # Nothing starts here. Run forward to the next position that
                # begins a dictionary word long enough to trust, and keep
                # everything before it as a single unknown token. The script
                # also changes character class at a real boundary, so Thai
                # text is never glued to digits or Latin letters.
                j, thai = i + 1, self._is_thai(run[i])
                while j < len(run):
                    if self._is_thai(run[j]) != thai:
                        break
                    nxt = self._longest_at(run, j, longest)
                    if (nxt and len(nxt) >= self.MIN_INTERNAL_MATCH
                            and j - i >= self.MIN_UNKNOWN_CHUNK):
                        break
                    j += 1
                words.append(run[i:j])
                i = j
        return words

    def _known(self, word: str) -> bool:
        return word in self._words

    def _resegment_unknown(self, tokens: List[str]) -> List[str]:
        """Split an unknown token into known pieces where that is possible.

        The segmenter works from a general Thai word list, so it happily
        returns compounds we have no entry for - "กินข้าว" ("eat rice") comes
        back whole and is reported as unknown even though BOTH halves are in
        the dictionary. Splitting it recovers a full translation instead of a
        pass-through. Only applied when EVERY piece is known, so a genuinely
        unknown word is never chopped into misleading fragments.
        """
        out: List[str] = []
        for word in tokens:
            if self._known(word) or len(word) < 4:
                out.append(word)
                continue
            pieces, i, ok = [], 0, True
            while i < len(word):
                for size in range(len(word) - i, 0, -1):
                    piece = word[i:i + size]
                    if self._known(piece) and size >= 2:
                        pieces.append(piece)
                        i += size
                        break
                else:
                    ok = False
                    break
            out.extend(pieces if (ok and len(pieces) > 1) else [word])
        return out

    def tag(self, tokens: List[str]) -> List[str]:
        if not _PYTHAINLP:
            return [""] * len(tokens)
        try:
            return [t for _, t in pos_tag(tokens, corpus="orchid")]
        except Exception:
            return [""] * len(tokens)

    # ------------------------------------------------------------------
    def classify(self, tokens: List[str], tags: List[str]) -> List[Token]:
        """Decide what each source token IS, before any assembly happens."""
        out: List[Token] = []
        for idx, (word, tag) in enumerate(zip(tokens, tags)):
            t = Token(thai=word, tag=tag, pos=G.orchid_to_pos(tag))
            prev = out[-1] if out else None

            if word in self.names:
                t.role, t.karen = "name", word
                t.note = "proper name, copied through untranslated"
            elif word in G.POLITENESS_WORDS:
                t.role, t.note = "dropped", "Thai politeness particle; no S'gaw Karen equivalent in any source"
            elif word in G.NEGATION_WORDS:
                t.role, t.karen = "grammar", "NEG"
            elif word in G.FUTURE_WORDS:
                t.role, t.karen = "grammar", "FUT"
            elif word in G.PERFECT_WORDS:
                t.role, t.karen = "grammar", "PERF"
            elif word in G.INTENSIFIER_WORDS:
                t.role, t.karen = "grammar", G.INTENSIFIER
            elif word in G.DEMONSTRATIVE_WORDS:
                t.role, t.karen = "grammar", G.DEMONSTRATIVE
            elif word in G.QUESTION_WORDS:
                t.role, t.karen = "question", G.QUESTION_WORDS[word]
            elif word in G.UNSOURCED_QUESTION_WORDS:
                t.role = "unknown"
                t.note = "question word with no S'gaw Karen script form in the dictionary; not guessed"
            elif word in G.POLAR_QUESTION_WORDS:
                t.role, t.karen = "dropped", ""
                t.note = ("yes/no question particle: the Grammar sheet documents a Karen polar "
                          "question marker phonemically (IPA /ha/) but no script form exists in "
                          "any source, so the question cannot be marked")
            elif word in G.THAI_NUMBERS:
                t.role, t.karen = "number", G.THAI_NUMBERS[word]
            elif word.isdigit():
                t.role, t.karen = "number", "".join(G.ARABIC_TO_KAREN.get(c, c) for c in word)
            elif all(c in G.THAI_DIGIT_TO_KAREN for c in word):
                t.role, t.karen = "number", "".join(G.THAI_DIGIT_TO_KAREN[c] for c in word)
            elif prev is not None and prev.role == "number" and word in G.THAI_CLASSIFIERS:
                # A classifier is only a classifier right after a number.
                # ตัว, ใบ, ที่ and others are ordinary words otherwise, and
                # looking them up as nouns there was producing junk
                # ("four cat BODY" instead of "four cats").
                t.role, t.karen = "classifier", G.CLASSIFIER
                t.note = ("Thai marks the noun class here; Karen has only one general "
                          "classifier in our sources, so the distinction is lost")
            elif word in CORE:
                karen, pos, conf, note = CORE[word]
                t.role, t.karen, t.pos = "core", karen, pos or t.pos
                t.sense = Sense(thai=word, karen=karen, pos=pos, confidence=conf,
                                source="hand-verified core lexicon", category="")
                t.note = note
            elif word in G.LOCATIVE_WORDS:
                t.role, t.karen = "grammar", G.LOCATIVE
            elif word in G.GO_WORDS and word not in self.lex:
                t.role, t.karen = "grammar", G.GO
            else:
                sense = self.lex.lookup(word, t.pos)
                if sense is not None:
                    t.role, t.karen, t.sense = "lexicon", sense.karen, sense
                elif word in G.THAI_CLASSIFIERS:
                    # Only treated as a classifier if it is NOT a known noun -
                    # ตัว, ใบ and others are ordinary words too.
                    t.role, t.karen = "classifier", G.CLASSIFIER
                    t.note = "Thai has per-noun classifiers; Karen has only one general classifier in our sources"
                elif word in G.GO_WORDS:
                    t.role, t.karen = "grammar", G.GO
                else:
                    t.role = "unknown"
                    t.note = "not in the dictionary; passed through untranslated"
            out.append(t)
        return out

    # ------------------------------------------------------------------
    def assemble(self, tokens: List[Token]) -> Tuple[str, List[str]]:
        """Apply the transfer rules and build the Karen sentence.

        The two languages are both SVO with the same noun-phrase order, so
        this is close to monotone. The work is in the insertions Karen needs
        and Thai does not.
        """
        out: List[str] = []
        raw: List[bool] = []     # True where out[i] is untranslated Thai
        notes: List[str] = []

        def emit(text: str, passthrough: bool = False) -> None:
            out.append(text)
            raw.append(passthrough)

        pending_neg = False
        pending_fut = False
        clause_final: List[str] = []
        saw_question = False

        for i, t in enumerate(tokens):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None

            if t.role == "dropped":
                if t.note:
                    notes.append(f"dropped '{t.thai}': {t.note}")
                continue

            if t.role == "grammar" and t.karen == "NEG":
                pending_neg = True
                continue
            if t.role == "grammar" and t.karen == "FUT":
                pending_fut = True
                continue
            if t.role == "grammar" and t.karen == "PERF":
                clause_final.append(G.PERFECT)
                continue

            # A verb is where the negation / future markers attach.
            is_verb = (t.pos == "v") or (t.sense is not None and t.sense.pos in ("v", "adj"))
            if (pending_neg or pending_fut) and (is_verb or t.role in ("lexicon", "core", "grammar")):
                if pending_neg and pending_fut:
                    # Both particles are documented as sitting "immediately
                    # before the verb" and NO source resolves which comes
                    # first. Rather than invent an order, the negation is
                    # applied (it is the meaning-bearing one - dropping it
                    # would reverse the sentence) and the future is reported
                    # as unmarked. Found by testing "เขาจะไม่ไปโรงเรียน",
                    # where the future marker had been drifting past the verb
                    # and attaching to the following noun.
                    notes.append(
                        "negated future: Karen puts both the negation particle and the "
                        "future prefix immediately before the verb and no source in this "
                        "project resolves their order, so only the negation is marked here "
                        "- the future sense is NOT expressed"
                    )
                    pending_fut = False
                if pending_neg:
                    if t.thai in ALREADY_NEGATIVE:
                        # e.g. ช้า, whose Karen form တခ့ျဘၣ် is itself
                        # NEG+quick+NEG. Adding the circumfix again would
                        # produce a double negative in Karen.
                        notes.append(
                            f"'{t.thai}' is already negative in Karen "
                            f"({t.karen}); the negation circumfix was not added again"
                        )
                    else:
                        emit(G.NEG_PRE)
                        clause_final.append(G.NEG_POST)
                    pending_neg = False
                if pending_fut:
                    emit(G.FUTURE)
                    pending_fut = False

            if t.role == "question":
                saw_question = True

            emit(t.karen or t.thai, passthrough=t.role == "unknown")

            # "ไป <place-noun>" needs the allative particle Karen requires and
            # Thai does not. Skipped before a question word, which is already
            # locative ("where"), and before an explicit locative preposition.
            if t.karen == G.GO and nxt is not None:
                if nxt.role in ("lexicon", "core", "name") and nxt.karen and nxt.thai not in G.LOCATIVE_WORDS:
                    emit(G.ALLATIVE)

        if pending_neg:
            # Negation with nothing to attach to - keep it visible rather than
            # silently dropping a meaning-bearing particle.
            emit(G.NEG_PRE)
            clause_final.append(G.NEG_POST)
            notes.append("negation had no verb to attach to; placed before the last element")

        # Clause-final ordering: perfect first, then the negation partner, then
        # the question marker is already in place (wh-words are fused with it).
        ordered = [p for p in clause_final if p == G.PERFECT] + [p for p in clause_final if p == G.NEG_POST]
        for particle in ordered:
            emit(particle)

        if saw_question and G.PERFECT in ordered:
            notes.append("question combined with perfect aspect; particle order not confirmed by any source")

        return self._join(out, raw), notes

    @staticmethod
    def _join(parts: List[str], raw: List[bool]) -> str:
        """Space-separate the Karen words, but NOT untranslated Thai.

        Thai is written without spaces between words, so a stretch of Thai
        the dictionary could not cover is one unbroken piece of text to the
        reader. Spacing it out made a single unknown word look like several
        ("ส ว ั ส ด ี" for "สวัสดี"), which is both unreadable and a
        misreport of what the translator actually failed on.
        """
        pieces: List[str] = []
        prev_raw = False
        for text, passthrough in zip(parts, raw):
            if not text:
                continue
            if passthrough and prev_raw and pieces:
                pieces[-1] += text
            else:
                pieces.append(text)
            prev_raw = passthrough
        return " ".join(pieces)

    # ------------------------------------------------------------------
    def translate(self, text: str) -> Translation:
        text = " ".join(text.split())
        if not text:
            return Translation(source=text, karen="")
        raw = self.segment(text)
        raw = self._resegment_unknown(raw)
        tags = self.tag(raw)
        tokens = self.classify(raw, tags)
        karen, notes = self.assemble(tokens)

        unknown = [t.thai for t in tokens if t.role == "unknown"]
        if unknown:
            notes.append("not in the dictionary, passed through as Thai: " + ", ".join(unknown))
        return Translation(source=text, karen=karen, tokens=tokens, notes=notes)
