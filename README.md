# KSR2TH

Rule-based bidirectional translator for Thai ↔ S'gaw Karen. No model, no training:
tokens are resolved against a dictionary and assembled by documented transfer rules,
and every output token can be traced to the row or rule that produced it.

```
$ python rule_translate.py --text "ผมไม่ชอบกาแฟ"
TH  ผมไม่ชอบกาแฟ
KR  ယ တ ဘၣ်သး ခီဖံၣ် ဘၣ်
    coverage 100%   lowest source confidence: M
```

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Design notes](#design-notes)
- [Reverse direction: S'gaw Karen → Thai](#reverse-direction-sgaw-karen--thai)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Extending](#extending)
- [License](#license)

## Requirements

- Python 3.8+
- `pythainlp` — optional but recommended. Provides Thai segmentation, POS tagging, and
  the TNC frequency table used to rank Karen→Thai candidates. Without it the translator
  falls back to a dictionary-driven longest-match segmenter: same pipeline, weaker word
  boundaries.
- `openpyxl` — only for `build_lexicon.py`.
- `sacrebleu` — only for `evaluate_rules.py`.

## Installation

```bash
git clone https://github.com/<user>/KSR2TH.git
cd KSR2TH
pip install -r requirements.txt
```

`rules/lexicon.csv` is committed, so the translator runs immediately after cloning.
Rebuild it only when the source workbook changes:

```bash
python build_lexicon.py Sgaw_Karen_English_Thai_Dictionary2.xlsx
```

## Usage

### Command line

```bash
python rule_translate.py --text "ผมไม่ชอบกาแฟ"                       # Thai → Karen
python rule_translate.py --direction kr2th --text "ယ တ ဘၣ်သး ခီဖံၣ် ဘၣ်"  # Karen → Thai
python rule_translate.py --text "ผมไม่ชอบกาแฟ" --explain              # per-token report
python rule_translate.py --interactive
python rule_translate.py --file sentences.txt --json out.json
```

| Flag | Default | Description |
|---|---|---|
| `--direction {th2kr,kr2th}` | `th2kr` | Translation direction. |
| `--text TEXT` | — | Translate a single sentence. |
| `--file PATH` | — | Translate each line of a file. |
| `--interactive` | — | Read sentences from stdin until `quit`. |
| `--explain` | off | Print the decision for every token: source role, confidence, notes. |
| `--json PATH` | — | Write results as JSON. |

One of `--text`, `--file` or `--interactive` is required.

`--explain` output:

```
$ python rule_translate.py --text "ผมไม่ชอบกาแฟ" --explain
TH  ผมไม่ชอบกาแฟ
KR  ယ တ ဘၣ်သး ခီဖံၣ် ဘၣ်
    coverage 100%   lowest source confidence: M
    ------------------------------------------------------------------
    ผม             -> ယ                      core lexicon (hand-verified) [H]
                      1SG; overrides the workbook's 'hair' senses
    ไม่            -> NEG                    grammar rule
    ชอบ            -> ဘၣ်သး                  core lexicon (hand-verified) [H]
                      like; from the 'fond' entry, the 'like' row is empty
    กาแฟ           -> ခီဖံၣ်                 core lexicon (hand-verified) [M]
```

### Python API

```python
from rules.translator import RuleTranslator
from rules.reverse import KarenToThai

rt = RuleTranslator()
result = rt.translate("เขาจะไม่ไปโรงเรียน")

result.karen              # 'အဝဲပိၥ်ခွါ တ လဲၤ ဆူ ကၠိ ဘၣ်'
result.coverage           # 1.0  — share of content tokens translated
result.unknown_words      # []   — source tokens with no dictionary entry
result.lowest_confidence  # 'M'  — worst source rating used (H/M/L)
result.notes              # rule decisions and dropped-token explanations
result.tokens             # [Token(thai=..., karen=..., role=..., sense=...), ...]

kt = KarenToThai(rt.lex)  # pass the loaded lexicon to avoid reparsing the CSV
back = kt.translate(result.karen)
back.thai                 # 'เขาไม่ไปโรงเรียน'
back.ambiguous            # [(karen_word, [thai_alternates]), ...]
```

`Token.role` identifies the provenance of each token: `core`, `lexicon`, `grammar`,
`question`, `number`, `classifier`, `name`, `dropped`, or `unknown`.

## How it works

```
segment → POS tag → look up → apply transfer rules → assemble
```

1. **Segment.** Thai is written without spaces. `pythainlp`'s `newmm` tokenizer is seeded
   with the project's own headwords so multi-word entries survive as single units. The
   fallback matcher uses the same word set.
2. **Tag.** `pythainlp` ORCHID tags, mapped to the lexicon's POS codes.
3. **Look up.** The hand-verified core lexicon is consulted first, then the bulk lexicon
   with POS-aware sense selection. Function words bypass both and go to the grammar rules.
4. **Apply transfer rules.** Insertions Karen requires and Thai does not: the negation
   circumfix, the allative after a motion verb, clause-final aspect particles.
5. **Assemble.** Karen tokens joined with spaces; untranslated Thai joined without, as
   Thai is written.

### Why a transfer approach fits this pair

Thai and S'gaw Karen agree on most of the word order that usually makes rule-based
translation hard:

| | Thai | S'gaw Karen |
|---|---|---|
| clause | Subject-Verb-Object | Subject-Verb-Object |
| modified noun | noun + adjective | noun + stative verb |
| counting | noun + numeral + classifier | noun + numeral + classifier |
| place | preposition + place | preposition + place |
| verbs | no inflection | no inflection |

Translation is therefore close to monotone — mostly word-for-word, with a small number of
insertions, which is the case transfer rules handle well. The genuine differences are
implemented as rules:

- **Negation is a circumfix.** Thai uses one pre-verbal word; Karen needs `တ` before the
  verb *and* `ဘၣ်` closing the clause, a particle with no Thai counterpart.
- **`ไป <place>` takes an allative.** `ဆူ` is inserted before the place noun.
- **Thai politeness particles** (ครับ, ค่ะ, นะ …) have no Karen equivalent. They are dropped
  and reported, not guessed at.

## Repository layout

```
rule_translate.py        CLI, both directions
build_lexicon.py         builds rules/lexicon.csv from the dictionary workbook
evaluate_rules.py        BLEU/chrF plus coverage metrics
outputs_rules_eval.json  last recorded evaluation run
rules/
├── lexicon.csv          3,446 Thai headwords · 6,871 senses
├── lexicon.py           lookup, POS-aware sense selection, provenance per entry
├── core_lexicon.py      ~120 hand-verified high-frequency words, override the bulk data
├── grammar.py           Karen particles, Thai function words, transfer rules
├── names.txt            proper names copied through untranslated — editable
├── translator.py        Thai → Karen pipeline
├── reverse_lexicon.py   Karen → Thai index, candidates ranked by Thai word frequency
└── reverse.py           Karen → Thai segmentation, sense choice, rule inversion
```

## Design notes

### The core lexicon overrides the bulk data on common words

The bulk lexicon fails on exactly the words that carry the grammar. Found by testing:

```
ผม  → "hair"          Thai ผม is both "I" and "hair"; every workbook sense is the
                      noun, so the pronoun reading is not there to be found
มี  → ပၥ်ဃုၥ်          four senses all rated H, plain "have" (အိၣ်ဒီး) is fourth;
                      confidence order cannot break the tie
ไป  → လၢကျဲ ဒီဘိ       the workbook's ไป rows are all adverbial ("along the way");
                      the plain verb "go" is not among them
```

~120 words are pinned in `core_lexicon.py` and consulted before the bulk lexicon. Each is
the same hand-verified mapping used by the companion neural project's `dataset/templates.py`,
whose `KAREN_WORD_SOURCES` table records the English headword, confidence rating, and
source codes.

### Unknown words are passed through and flagged

```
TH  สวัสดี วันนี้อากาศเป็นอย่างไรบ้าง
KR  สวัสดี တနံၤအံၤ မူခိၣ်ကလံၤသီၣ်ဂီၤ မ့ၢ် อย่างไรบ้าง
    coverage 50%
    ! not in the dictionary, passed through as Thai: สวัสดี, อย่างไร, บ้าง
```

Untranslated Thai is emitted without internal spaces, so an unknown word reads as one
word; the notes list the missing items separately. Each missing word is a dictionary row
to add, and adding it takes effect immediately with no retraining.

### Bugs found by testing, and how they are handled

1. **`ไม่ชอบ` glued into one lexical entry.** The workbook has "ไม่ชอบ" (dislike) as a
   headword, so the segmenter kept it whole, negation resolved *lexically*, and the
   clause-final `ဘၣ်` was never emitted. ไม่-/จะ- prefixed headwords are now removed from
   the segmenter's word list, forcing negation through the grammar rules.
2. **Negated future placed the marker wrongly.** `เขาจะไม่ไปโรงเรียน` produced
   `... တ လဲၤ ဆူ က ကၠိ ...` — the future prefix drifted past the verb onto the following
   noun. Both particles are documented as sitting immediately before the verb and no
   source resolves their order, so the negation is applied (dropping it would reverse the
   sentence) and the unmarked future is reported in the notes.
3. **Numbered sense lists leaked into output.** Karen cells such as
   `၁ ကွၢ်ဃု, ဃုထံၣ် ၂ ထံၣ်န့ၢ်က့ၤ` are numbered sense lists; splitting on commas alone left
   the digits in, so `ตก` translated to `၁ လဲၤလီၤ ၂ လီၤစၢၤ`. Only the first sense is kept.
4. **Proper names split and half-translated.** `สมชาย` segments into `สม` + `ชาย`, and
   `ชาย` is a real word ("male"), so half the name was translated — an error that is hard
   to spot. `rules/names.txt` holds names to copy through; add to it when a new name
   appears. Thai has no spaces and nothing marks a name as a name, so a list is unavoidable.
5. **Unknown words were shredded into single characters.** Without `pythainlp`, the
   fallback segmenter emitted one character at a time whenever nothing matched, so `สวัสดี`
   — a greeting with no bare headword in the workbook — came out as six tokens
   `ส ว ั ส ด ี`, and a lone vowel mark is not a word. Uncovered characters are now kept
   together as one token: the matcher runs forward to the next position that starts a
   dictionary word of at least three characters. The three-character floor stops `ดี`
   ("good") from cutting the greeting in half; a one-character floor on the unknown piece
   itself stops `ย่าง` from leaving a bare `อ` behind in `อย่างไรบ้าง`.
6. **The fallback segmenter did not know the function words.** `_known()` consulted only
   the dictionary while the `pythainlp` path was separately seeded with the grammar's
   particle lists, so without `pythainlp`, `จะ` was not a word: `เขาจะไปโรงเรียนพรุ่งนี้`
   produced an unknown `จะไป` instead of a future marker and a verb, and `ไม่ชอบ` survived
   whole — reintroducing bug 1 on that path. Both paths now build from one word set
   (`_segmenter_words`).

Also handled: `กินข้าว` returned as one unknown compound although both halves are in the
dictionary. Unknown tokens are re-split against known words, but only when *every* piece
is known, so a genuinely unknown word is never chopped into misleading fragments.

## Reverse direction: S'gaw Karen → Thai

```bash
python rule_translate.py --direction kr2th --text "ယ တ ဘၣ်သး ခီဖံၣ် ဘၣ်"
python rule_translate.py --direction kr2th --interactive
```

```
KR  ယ တ ဘၣ်သး ခီဖံၣ် ဘၣ်                      TH  ผมไม่ชอบกาแฟ
KR  အဝဲပိၥ်ခွါ က လဲၤ ဆူ ကၠိ ခဲမုၢ်ဆ့ၣ်        TH  เขาจะไปโรงเรียนพรุ่งนี้
KR  အဝဲပိၥ်မုၣ် တ အိၣ်ဒီး သိလ့ၣ် ၂ ဖျၢၣ် ဘၣ်  TH  หล่อนไม่มีรถยนต์สองคัน
KR  န လဲၤ ဖဲလဲၣ်                                TH  คุณไปที่ไหน
```

This is not the forward translator run backwards. Three things differ, each with its own
implementation.

**No segmenter exists for Karen.** There is no `pythainlp` equivalent — no tokenizer, no
tagger, no corpus. Karen is written with spaces, but they mark phrase boundaries as much as
word boundaries, and a multi-word dictionary entry can span them. Segmentation is therefore
dictionary-driven: whitespace chunks are tried in groups of three, then two, then one, and
only then is a chunk broken up internally by longest match. Unmatched text is kept as one
unknown piece rather than fragmented into separately-reported missing words.

**Ambiguity is much heavier in this direction.** One Karen word maps back to several Thai
words for 513 of 6,259 headwords (8.2%), and the workbook gives no ordering among them —
they came from different rows. Candidates are ranked by Thai word frequency, using the Thai
National Corpus list bundled offline with `pythainlp`.

The ranking order was determined by testing. Sorting by confidence first looked principled
but picked `สถาบัน` ("institution", rated H) over `โรงเรียน` ("school", rated M) for `ကၠိ` —
wrong for nearly every sentence. H and M are both "attested"; between two attested synonyms
the useful signal is which one Thai speakers actually use. So only L is demoted, and
frequency decides among the rest. Alternates are always reported:

```
! 'ကၠိ' could also be สถาบัน - chose 'โรงเรียน' as the commonest Thai word
```

**The grammar rules invert unevenly.** Karen's negation circumfix collapses into a single
Thai word, so the clause-final `ဘၣ်` must be located and deleted — requiring a look-ahead to
the end of the clause, not just the current token. Karen's allative `ဆူ` disappears entirely
(Thai is `ไปโรงเรียน`, no preposition). Karen's single general classifier must be expanded
back into the specific Thai classifier, which is information Karen does not carry: it is
recovered from the noun where known (`รถยนต์` → `คัน`) and falls back to the generic `อัน`
with a note where not.

## Evaluation

`evaluate_rules.py` requires `sacrebleu` and the `generated_dataset/` splits from the
companion neural project, which are not part of this repository. Figures below are the last
recorded run, stored in `outputs_rules_eval.json` and measured with `pythainlp` installed.

### Thai → S'gaw Karen

| | `test.jsonl` (template grammar) | `test_unseen_vocab.jsonl` |
|---|---|---|
| BLEU | 58.9 | 73.0 |
| chrF | 77.6 | 83.8 |
| exact match | 50.1% | 66.6% |
| dictionary coverage | 90.9% | 97.9% |
| fully-covered sentences | 68.6% | 93.6% |

### S'gaw Karen → Thai, and round trip

| | `test.jsonl` | `test_unseen_vocab.jsonl` |
|---|---|---|
| kr2th chrF | 72.0 | 84.1 |
| kr2th exact match | 40.5% | 64.8% |
| dictionary coverage | 97.4% | 98.5% |
| round trip th→kr→th, chrF | 87.0 | 89.1 |
| round trip exact match | 55.8% | 72.0% |

Read chrF rather than BLEU whenever Thai is on the output side: Thai is written without
spaces, which makes word-level BLEU close to meaningless.

### Reading the two splits

`test_unseen_vocab.jsonl` is built from words held out of neural training entirely. A rule
engine has no train/test distinction, so held-out words are ordinary dictionary words to it;
a neural model cannot emit a word absent from its training data. That asymmetry is why the
second column is the more informative one.

`test.jsonl` is generated by the same templates the neural model trained on, so it measures
recall of memorised sentences for that model (BLEU 95.9) and flatters both systems through
overlapping vocabulary. It also contains deliberate typos, which dictionary lookup handles
worse than a model does.

The round trip passes through both rule sets, so every lossy step compounds: Karen carries
no politeness particles and only one classifier, so some Thai detail cannot survive by
construction. chrF 87–89 with over half the sentences returning character-identical is a
floor on quality, not a measure of either direction alone.

## Limitations

This is a glossing engine with real grammar, not a fluent translator. It is designed so that
a visible gap is preferable to a confident wrong sentence.

**Both directions**

- Word-by-word translation with local rules. There is no model of meaning; ambiguity is not
  resolved from context beyond part of speech.
- Long or complex sentences — relative clauses, coordination, embedded speech — pass through
  largely word-for-word. Usually understandable, not fluent.
- Coverage is bounded by the dictionary: 3,446 Thai headwords. Ordinary Thai has far more.
- Roughly one in four BFAS-APK-sourced entries may carry a character-level error (~77%
  decode accuracy, per the project notes). Every output row carries its source confidence.

**Thai → Karen**

- Yes/no questions cannot be marked. The Grammar sheet documents a Karen polar-question
  particle phonemically (IPA /ha/), but no script form exists in the workbook's 22,543 rows.
  The particle is dropped and the sentence flagged.
- Without `pythainlp`, word boundaries are weaker. A short dictionary word at the start of an
  unknown one still wins: `เทคโนโลยี` ("technology") begins with `เท`, a headword, so it
  segments as `เท` + unknown `คโนโลยี`.

**Karen → Thai**

- No POS tagger exists for Karen, so sense selection has less to work with than the forward
  direction, where `pythainlp`'s tagger narrows the candidates.
- Thai classifiers cannot always be recovered, because Karen does not encode which one.
  Extend `THAI_CLASSIFIER_BY_NOUN` in `rules/reverse.py` to cover more nouns.
- Politeness is unrecoverable. Thai ครับ/ค่ะ are dropped going out and nothing in the Karen
  brings them back, so translations read blunt by Thai conventions.

## Extending

| Goal | Where |
|---|---|
| Add or correct a translation | `rules/lexicon.csv`, or `rules/core_lexicon.py` to override the bulk data |
| Add a proper name | `rules/names.txt`, one per line, `#` for comments |
| Add a grammar rule or particle | `rules/grammar.py`, then the relevant branch in `classify()` / `assemble()` |
| Recover more Thai classifiers | `THAI_CLASSIFIER_BY_NOUN` in `rules/reverse.py` |
| Rebuild from an updated workbook | `python build_lexicon.py <workbook>.xlsx` |

Every lexicon row carries `confidence` and `source` columns, and `--explain` reports both,
so a questionable output can be traced to the row behind it.

## License

[GNU AGPL-3.0-or-later](https://www.gnu.org/licenses/agpl-3.0.html) — see [`LICENSE`](LICENSE)
and [`COPYRIGHT`](COPYRIGHT).

Copyright (C) 2026 Wattanachai Maijampa

```
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.
```

In practice:

- Use, study, modify and redistribute it freely, including commercially.
- Derivatives must stay free software under AGPL-3.0: anyone distributing a modified version
  must publish their source under the same license.
- Network use is covered (AGPL §13). If you deploy a Thai–Karen translation site or API built
  on this, its users must be offered the source of *your* version — typically a "Source" link
  in the interface. A free-of-charge service is covered exactly like a paid one; running an
  unmodified copy triggers no extra obligation.

### Dictionary data

`rules/lexicon.csv` is derived from a compiled S'gaw Karen dictionary workbook whose rows
carry their own source codes (`MOBI`, `BFAS-APK`, `PICDICT95`, `KTWG`, `DRUM-WL`, `JSS-COLOR`,
`MNKOM`, `WIKI-LANG`). The license above covers this code and the work of compiling the data;
it conveys no rights over the underlying lexicographic sources, which belong to their
respective authors. Every row keeps its `source` column so any entry can be traced back.
