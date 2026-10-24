# KSR2TH - S'gaw Karen Translator Using a Rule-Based
# Copyright (C) 2026 Wattanachai Maijampa
# SPDX-License-Identifier: AGPL-3.0-or-later
"""S'gaw Karen grammar: particles, Thai function words, and the transfer rules.

Every Karen particle here comes from the dictionary workbook's `Grammar` sheet
(sourced from Atsuhiko Kato's academic description of Karen plus Wikipedia's
cited summary), and each is recorded with where it came from.

WHY A TRANSFER APPROACH WORKS WELL FOR THIS PAIR
------------------------------------------------
Thai and S'gaw Karen agree on most of the word order that usually makes
rule-based translation hard:

    Thai                          S'gaw Karen
    Subject-Verb-Object           Subject-Verb-Object      (same)
    Noun + adjective              Noun + stative verb      (same)
    Noun + numeral + classifier   Noun + numeral + CLF     (same)
    preposition + place           preposition + place      (same)
    no verb inflection            no verb inflection       (same)

So the translation is close to monotone: mostly word-for-word, with a small
number of INSERTION rules. That is exactly the situation transfer rules handle
well, and it is why this approach can beat a neural model trained on a corpus
that has no real sentences in it.

The real differences are handled below:
  * Thai marks negation with ONE pre-verbal word; Karen uses a CIRCUMFIX, so a
    clause-final particle has to be inserted that has no Thai counterpart.
  * Thai "ไป <place>" needs an allative particle inserted in Karen.
  * Thai politeness particles have no Karen equivalent and are dropped.
"""
from __future__ import annotations

from typing import Dict, Tuple

# -----------------------------------------------------------------------------
# Karen grammatical particles - all from the workbook's Grammar sheet
# -----------------------------------------------------------------------------
NEG_PRE = "တ"        # immediately before the verb        (Grammar: Negation)
NEG_POST = "ဘၣ်"     # closes the clause                  (Grammar: Negation)
FUTURE = "က"         # prospective prefix                 (Grammar: Prospective aspect)
PERFECT = "လံ"       # clause-final perfect               (Grammar: Tense / aspect)
INTENSIFIER = "မး"   # "very / a lot"                     (dictionary: very, H)
CLASSIFIER = "ဖျၢၣ်"  # the one general classifier         (Grammar: Classifiers)
DEMONSTRATIVE = "အံၤ"  # postnominal "this"                (Grammar: Demonstratives)
LOCATIVE = "ဖဲ"      # "at"                               (dictionary: at, H)
ALLATIVE = "ဆူ"      # "to"                               (dictionary: to, H)
GO = "လဲၤ"           # "go"                               (dictionary: go, M)

PARTICLE_SOURCES: Dict[str, str] = {
    NEG_PRE: "Grammar sheet 'Negation' (KATO-KAREN;WIKI-LANG); cross-confirmed by the dictionary's 'not' entry, which is NEG+copula+NEG",
    NEG_POST: "Grammar sheet 'Negation' (KATO-KAREN;WIKI-LANG); same cross-confirmation",
    FUTURE: "Grammar sheet 'Prospective aspect' (WIKI-LANG)",
    PERFECT: "Grammar sheet 'Tense / aspect' (WIKI-LANG)",
    INTENSIFIER: "dictionary 'very' (H, BFAS-APK;MOBI)",
    CLASSIFIER: "Grammar sheet 'Classifiers & measure words' (WIKI-LANG)",
    DEMONSTRATIVE: "Grammar sheet 'Demonstratives' (WIKI-LANG)",
    LOCATIVE: "dictionary 'at' (H, BFAS-APK;MOBI)",
    ALLATIVE: "dictionary 'to' (H, BFAS-APK;MOBI)",
    GO: "dictionary 'go' (M, DRUM-WL;MOBI)",
}

# -----------------------------------------------------------------------------
# Thai function words, handled as GRAMMAR rather than looked up in the lexicon
# -----------------------------------------------------------------------------
NEGATION_WORDS = {"ไม่", "มิ", "บ่", "ไม่ได้", "มิได้"}
FUTURE_WORDS = {"จะ"}
PERFECT_WORDS = {"แล้ว", "เเล้ว"}
INTENSIFIER_WORDS = {"มาก", "มากๆ", "มาก ๆ", "เยอะ"}
DEMONSTRATIVE_WORDS = {"นี้", "นี่"}
LOCATIVE_WORDS = {"ที่", "ณ"}
GO_WORDS = {"ไป"}

# Politeness / hedging particles with no S'gaw Karen equivalent in any source.
# Dropped rather than guessed at, and reported in the translation notes so the
# omission is visible instead of silent.
POLITENESS_WORDS = {"ครับ", "ค่ะ", "คะ", "ครับผม", "นะ", "น่ะ", "จ้ะ", "จ๊ะ",
                    "สิ", "เลย", "ล่ะ", "หรอก", "ด้วย", "หน่อย"}

# Thai numerals -> Karen digits. The dictionary supplies Karen DIGITS at High
# confidence but has no spelled-out Karen number words, so digits are used.
THAI_NUMBERS: Dict[str, str] = {
    "ศูนย์": "၀", "หนึ่ง": "၁", "สอง": "၂", "สาม": "၃", "สี่": "၄", "ห้า": "၅",
    "หก": "၆", "เจ็ด": "၇", "แปด": "၈", "เก้า": "၉", "สิบ": "၁၀",
    "ยี่สิบ": "၂၀", "สามสิบ": "၃၀", "สี่สิบ": "၄၀", "ห้าสิบ": "၅၀",
    "หกสิบ": "၆၀", "เจ็ดสิบ": "၇၀", "แปดสิบ": "၈၀", "เก้าสิบ": "၉၀",
    "ร้อย": "၁၀၀", "หนึ่งร้อย": "၁၀၀", "พัน": "၁၀၀၀", "หนึ่งพัน": "၁၀၀၀",
}
ARABIC_TO_KAREN = {str(i): d for i, d in enumerate("၀၁၂၃၄၅၆၇၈၉")}
THAI_DIGIT_TO_KAREN = {d: k for d, k in zip("๐๑๒๓๔๕๖๗๘๙", "၀၁၂၃၄၅၆၇၈၉")}

# Thai classifiers. Thai distinguishes these; Karen has only the one general
# classifier in any of this project's sources, so they all map to it. The
# mapping is many-to-one and lossy IN THAT DIRECTION ONLY - nothing is invented.
THAI_CLASSIFIERS = {
    "ตัว", "คัน", "เล่ม", "ลูก", "แก้ว", "เครื่อง", "เรื่อง", "เพลง", "อัน",
    "ชิ้น", "คน", "ใบ", "ผล", "ต้น", "หลัง", "ดวง", "อย่าง", "ครั้ง", "แผ่น",
    "ก้อน", "ขวด", "ถ้วย", "จาน", "คู่", "ชุด", "องค์", "รูป", "ที่",
}

# Thai wh-words -> the FUSED Karen wh-word + clause-final question marker.
#
# The Grammar sheet gives the content-question marker phonemically as /lɛ̂/ with
# no script form, but every wh-word in the workbook's Question words category
# ends in လဲၣ်, which identifies it. The wh-word and marker are written fused,
# so each value here is used as a unit.
QUESTION_WORDS: Dict[str, str] = {
    "อะไร": "မနုၥ်လဲၣ်",
    "ที่ไหน": "ဖဲလဲၣ်",
    "ไหน": "ဖဲလဲၣ်",
    "ใคร": "မတၤလဲၣ်",
    "ทำไม": "ဘၣ်မနုၤလဲၣ်",
}
# Thai wh-words with NO sourced Karen form anywhere in the 22,543-row
# dictionary. Translated as unknown and flagged, never guessed.
UNSOURCED_QUESTION_WORDS = {"เมื่อไหร่", "อย่างไร", "ยังไง", "กี่", "เท่าไหร่", "เท่าไร"}

# Thai yes/no question particles. The Grammar sheet documents a Karen polar
# question particle phonemically (IPA /ha/) but NO script form for it exists in
# any source, so polar questions cannot be marked. Flagged, not guessed.
POLAR_QUESTION_WORDS = {"ไหม", "มั้ย", "หรือ", "รึ", "หรือเปล่า", "ใช่ไหม"}


def orchid_to_pos(tag: str) -> str:
    """Map a pythainlp ORCHID part-of-speech tag to our lexicon's POS codes."""
    if not tag:
        return ""
    if tag.startswith("NEG"):
        return "neg"
    if tag.startswith("ADV"):
        return "adv"
    if tag.startswith("VATT"):
        return "adj"       # attributive verb = adjective-like
    if tag.startswith("V"):
        return "v"
    if tag.startswith("PPR") or tag.startswith("PDM"):
        return "pron"
    if tag.startswith("N"):
        return "n"
    return ""
