# KSR2TH - S'gaw Karen Translator Using a Rule-Based
# Copyright (C) 2026 Wattanachai Maijampa
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hand-verified mappings for the most common Thai words.

WHY THIS EXISTS
---------------
The bulk lexicon is excellent for content words and unreliable for the few
hundred words that appear in almost every sentence. Three failures, all real
and all found by testing rather than by reasoning:

    ผม    -> "hair"        Thai ผม is both "I" and "hair"; every sense row in
                           the workbook is the noun, so the pronoun reading -
                           the one that matters - is not there to be found.
    มี    -> ပၥ်ဃုၥ်        four senses all rated H, and plain "have" (အိၣ်ဒီး)
                           is fourth in the list, so confidence order cannot
                           break the tie.
    ไป    -> လၢကျဲ ဒီဘိ     the workbook's ไป rows are all adverbial senses
                           ("along the way"); the plain verb "go" is not
                           among them.

A word-for-word translator lives or dies on these, because they carry the
grammar. So the highest-frequency words are pinned here, and this table is
consulted BEFORE the bulk lexicon.

Nothing here is invented. Every entry is the same hand-verified mapping used
by dataset/templates.py, whose KAREN_WORD_SOURCES table records the English
headword, the workbook's confidence rating and source codes for each one.
"""
from __future__ import annotations

from typing import Dict, Tuple

# thai -> (karen, pos, confidence, note)
CORE: Dict[str, Tuple[str, str, str, str]] = {
    # --- pronouns (Grammar sheet: Pronouns; 3rd person is gender-neutral,
    #     the gender word is kept only to preserve the Thai distinction) -----
    "ผม": ("ယ", "pron", "H", "1SG; overrides the workbook's 'hair' senses"),
    "ฉัน": ("ယ", "pron", "H", "1SG"),
    "ดิฉัน": ("ယ", "pron", "H", "1SG"),
    "กระผม": ("ယ", "pron", "H", "1SG"),
    "ข้าพเจ้า": ("ယ", "pron", "H", "1SG"),
    "คุณ": ("န", "pron", "H", "2SG"),
    "เธอ": ("န", "pron", "H", "2SG"),
    "ท่าน": ("န", "pron", "H", "2SG polite"),
    "เขา": ("အဝဲပိၥ်ခွါ", "pron", "H", "3SG; lit. 3SG + male"),
    "หล่อน": ("အဝဲပိၥ်မုၣ်", "pron", "H", "3SG; lit. 3SG + female"),
    "เธอผู้หญิง": ("အဝဲပိၥ်မုၣ်", "pron", "H", "3SG female"),
    "เรา": ("ပ", "pron", "L", "1PL"),
    "พวกเรา": ("ပ", "pron", "L", "1PL"),
    "พวกเขา": ("အဝဲသ့ၣ်", "pron", "M", "3PL"),
    "มัน": ("အဝဲ", "pron", "M", "3SG neutral"),

    # --- high-frequency verbs -------------------------------------------
    "ไป": ("လဲၤ", "v", "M", "go"),
    "มา": ("ဟဲ", "v", "M", "come"),
    "มี": ("အိၣ်ဒီး", "v", "L", "have; 4th of four H-rated senses, chosen by hand"),
    "เป็น": ("မ့ၢ်", "v", "M", "copula; confirmed by the 'not' entry, NEG+this+NEG"),
    "คือ": ("မ့ၢ်", "v", "M", "copula"),
    "อยู่": ("အိၣ်", "v", "M", "be located"),
    "ชอบ": ("ဘၣ်သး", "v", "H", "like; from the 'fond' entry, the 'like' row is empty"),
    "รัก": ("အဲၣ်", "v", "H", "love"),
    "เกลียด": ("သးဟ့", "v", "H", "hate"),
    "ต้องการ": ("အဲၣ်ဒိး", "v", "M", "want"),
    "อยาก": ("အဲၣ်ဒိး", "v", "M", "want"),
    "อยากได้": ("အဲၣ်ဒိး", "v", "M", "want"),
    "มองหา": ("ကွၢ်ဃု", "v", "H", "look for; from 'search'"),
    "หา": ("ကွၢ်ဃု", "v", "H", "look for; from 'search'"),
    "กิน": ("အီၣ်", "v", "H", "eat"),
    "ทาน": ("အီၣ်", "v", "H", "eat"),
    "ดื่ม": ("အီ", "v", "H", "drink"),
    "ช่วย": ("မၤစၢၤ", "v", "M", "help"),
    "รู้": ("သ့ၣ်ညါ", "v", "M", "know"),
    "ทราบ": ("သ့ၣ်ညါ", "v", "M", "know"),
    "เข้าใจ": ("နၢ်ပၢၢ်", "v", "H", "understand"),
    "พูด": ("ကတိၤတၢ်", "v", "M", "speak"),
    "ทำ": ("မၤ", "v", "M", "do/make"),
    "เห็น": ("ထံၣ်", "v", "M", "see"),
    "ให้": ("ဟ့ၣ်", "v", "M", "give"),

    # --- places ----------------------------------------------------------
    "ตลาด": ("ဖၠါ", "n", "H", "market"),
    "โรงเรียน": ("ကၠိ", "n", "M", "school"),
    "โรงพยาบาล": ("တၢ်ဆါဟံၣ်", "n", "M", "hospital; lit. sick-house"),
    "สนามบิน": ("ကဘီယူၤသန့", "n", "H", "airport"),
    "ร้านอาหาร": ("တၢ်အီၣ်ကျး", "n", "M", "restaurant; lit. food-shop"),
    "บ้าน": ("ဟံၣ်", "n", "H", "home; spelled with the Sgaw-exclusive U+1063"),
    "มหาวิทยาลัย": ("ဖၠၣ်စိမိၤ", "n", "H", "university"),
    "สวนสาธารณะ": ("ဖီကရၢၢ်", "n", "M", "park"),
    "สถานีรถไฟ": ("လ့ၣ်မ့ၣ်အူသန့", "n", "H", "train station"),
    "ร้าน": ("ကျး", "n", "H", "shop"),

    # --- common nouns -----------------------------------------------------
    "กาแฟ": ("ခီဖံၣ်", "n", "M", "coffee"),
    "ชา": ("လၣ်ဖး", "n", "L", "tea"),
    "หนังสือ": ("လံၥ်", "n", "M", "book"),
    "เพลง": ("တၢ်သးဝံၣ်", "n", "H", "song"),
    "ดนตรี": ("တၢ်သးဝံၣ်", "n", "H", "music; from 'song'"),
    "ภาพยนตร์": ("တၢ်ဂီၤမူ", "n", "H", "movie"),
    "หนัง": ("တၢ်ဂီၤမူ", "n", "H", "movie"),
    "สุนัข": ("ထွံၣ်", "n", "H", "dog"),
    "หมา": ("ထွံၣ်", "n", "H", "dog"),
    "แมว": ("သၣ်မံယီၤ", "n", "H", "cat"),
    "รถยนต์": ("သိလ့ၣ်", "n", "M", "car"),
    "รถ": ("သိလ့ၣ်", "n", "M", "car"),
    "ฟุตบอล": ("ဖျၢၣ်ထူ", "n", "H", "football"),
    "ข้าว": ("ဘု", "n", "M", "rice"),
    "น้ำ": ("ထံ", "n", "M", "water"),
    "อาหาร": ("တၢ်အီၣ်တၢ်အီ", "n", "H", "food"),
    "ครู": ("သရၣ်သိၣ်လိတၢ်", "n", "H", "teacher"),
    "อาจารย์": ("သရၣ်သိၣ်လိတၢ်", "n", "H", "teacher"),
    "นักเรียน": ("ကၠိဖိ", "n", "M", "student"),
    "ฝน": ("တၢ်စူၤ", "n", "M", "rain"),
    "อากาศ": ("မူခိၣ်ကလံၤသီၣ်ဂီၤ", "n", "H", "weather; the bulk lexicon returns 'wind' for this"),
    "ตก": ("လီၤ", "v", "M", "fall"),
    "ลม": ("ကလံၤ", "n", "L", "wind"),
    "ภูเขา": ("ကစၢၢ်", "n", "H", "mountain"),
    "ดวงจันทร์": ("လါ", "n", "H", "moon"),
    "ดาว": ("ဆၣ်", "n", "M", "star"),
    "เมฆ": ("တၢ်အၢၣ်", "n", "H", "cloud"),
    "ท้องฟ้า": ("မူကပိၥ်လိၤ", "n", "M", "sky"),
    "บาท": ("ဘး", "n", "H", "baht"),
    "ราคา": ("အၦ့ၤ", "n", "H", "price"),

    # --- time -------------------------------------------------------------
    "เมื่อวาน": ("မဟါတနံၤ", "time", "M", "yesterday"),
    "วันนี้": ("တနံၤအံၤ", "time", "M", "today"),
    "พรุ่งนี้": ("ခဲမုၢ်ဆ့ၣ်", "time", "M", "tomorrow"),
    "คืนนี้": ("တနၤအံၤ", "time", "M", "tonight"),
    "เดือน": ("လါ", "time", "H", "month"),
    "เช้า": ("ဂီၤခီ", "time", "H", "morning"),
    "ตอนเช้า": ("ဂီၤခီ", "time", "H", "morning"),
    "คืน": ("မုၢ်နၤခီ", "time", "H", "night"),
    "กลางคืน": ("မုၢ်နၤခီ", "time", "H", "night"),

    # --- adjectives (stative verbs in Karen) ------------------------------
    "สวย": ("ဃံလၤ", "adj", "L", "beautiful"),
    "ใหญ่": ("ဒိၣ်", "adj", "M", "big"),
    "เล็ก": ("ဆံး", "adj", "M", "small"),
    "ร้อน": ("ကိၢ်", "adj", "M", "hot"),
    "เย็น": ("ခုၣ်", "adj", "H", "cool; the 'cold' row has no Karen text"),
    "หนาว": ("ခုၣ်", "adj", "H", "cold; approximated by 'cool'"),
    "ใหม่": ("အသီ", "adj", "M", "new"),
    "เก่า": ("လီၤလံၤ", "adj", "H", "old (of things)"),
    "ถูก": ("အၦ့ၤဆံး", "adj", "H", "cheap"),
    "แพง": ("အၦ့ၤဒိၣ်", "adj", "H", "expensive"),
    "เร็ว": ("ခ့ျ", "adj", "H", "fast; from 'quick'"),
    "ช้า": ("တခ့ျဘၣ်", "adj", "M", "slow; itself a negated form, NEG+quick+NEG"),
    "ดี": ("ဂ့ၤ", "adj", "H", "good"),

    # --- courtesy ---------------------------------------------------------
    "ขอบคุณ": ("တၢ်ဘျုး", "phrase", "H", "thank you"),
    "ลาก่อน": ("လဲၤမုၥ်မု", "phrase", "L", "goodbye"),
    "ใช่": ("မ့ၢ်", "phrase", "M", "yes"),
}

# Thai words that are ALREADY negative on their own. If one of these is
# preceded by ไม่ the result would be a double negative, so the rule engine
# must not add the circumfix on top.
ALREADY_NEGATIVE = {"ช้า"}
