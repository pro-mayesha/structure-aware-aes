# PERSUADE 2.0 Audit Summary

## Dataset source

- Raw files: `persuade_corpus_2.0_train.csv, persuade_corpus_2.0_test.csv` under `data/persuade20/raw/`
- Combined annotation rows: **285,383**
- Unique essays: **25,994**
- Official split column: `competition_set` (`train` / `test`)

## Column identification

| Research field | Column | Notes |
|---|---|---|
| Essay ID | `essay_id` | Numeric ID; one essay spans many rows |
| Competition essay ID | `essay_id_comp` | Used in Feedback Prize 2022; may differ across splits for one essay |
| Full essay text | `full_text` | Repeated on every discourse row |
| Holistic essay score | `holistic_essay_score` | **Primary score (1–6)**; no alternate holistic column |
| Prompt | `prompt_name` | Short prompt label (15 unique) |
| Writing task | `task` | `Independent` or `Text dependent` (source-based) |
| Source text | `source_text` | Provided for text-dependent tasks |
| Assignment | `assignment` | Full writing prompt / instructions |
| Discourse label | `discourse_type` | Includes `Unannotated` gap segments |
| Discourse text | `discourse_text` | Segment text; may be trimmed vs raw offsets |
| Start / end positions | `discourse_start`, `discourse_end` | **Character offsets (0-indexed, inclusive end)** |
| Effectiveness | `discourse_effectiveness` | `Effective`, `Adequate`, `Ineffective`; null for `Unannotated` |
| Grade level | `grade_level` | 6, 8, 9, 10, 11, 12 |
| Demographics | `gender`, `ell_status`, `race_ethnicity`, `economically_disadvantaged`, `student_disability_status` | Essay-level metadata |

**Holistic score candidates:** Only `holistic_essay_score` exists. Values are integers 1–6 per PERSUADE 2.0 paper.

**Prompt candidates:** `prompt_name` (short label) and `assignment` (full text). Use `prompt_name` for grouping; `assignment` for task text.

## 1. Can holistic scores and discourse annotations be joined reliably?

**Yes, within a single long-format table.** Each row already joins essay text, holistic score, prompt, and one discourse segment. At essay level: 25,994/25,994 (100.00%) essays have full text, holistic score, and ≥1 annotated component. **25,992** essays (99.99%) pass strict join validation (consistent text/score, prompt present, annotations present).

- Duplicate essay IDs in essay table: 0
- Essays with conflicting `full_text` across rows: 2
- Essays appearing in both train and test: 1
- One-to-many structure: expected (mean 11.0 rows per essay)

## 2. How many essays are fully usable?

- **25,992** essays with `join_valid=True` (99.99%)
- Excluded primarily due to: {'conflicting_full_text': 1, 'conflicting_full_text;essay_in_multiple_splits': 1}

## 3. Are genuine discourse positions and labels available?

- **Labels present:** Claim, Concluding Statement, Counterclaim, Evidence, Lead, Position, Rebuttal
- **Additional label:** `Unannotated` (47,410 segments)
- **Unexpected labels:** none
- **Character offsets:** yes (`discourse_start`, `discourse_end`)
- Offset recovery: exact inclusive match 37,514; strip match 69,107; trimmed discourse_text 0; mismatch 118,429; outside boundary 836

Offsets are usable for structure reconstruction; prefer slicing `full_text` over trusting `discourse_text` when they disagree.

## 4. Are component-effectiveness labels available?

- **Yes** for annotated discourse types: Adequate, Effective, Ineffective
- Null for `Unannotated` segments (by design)

## 5. Are there serious annotation-quality problems?

- Partition covers full essay (inclusive spans): 1,512 essays
- Partition issues: {'does_not_end_at_last_char': 24482}
- Annotated span overlaps: 2 essays
- Outside-boundary annotated spans: 836 rows

Most essays have small trailing gaps between last span end and text length. `discourse_text` sometimes trims leading/trailing punctuation relative to offsets.

## 6. Can we construct an ordered structured-document representation?

**Yes.** Sort segments by `discourse_start` (not `discourse_type_num`, which does not reflect document order). Heuristic discourse-order validity: 22,902/25,994 essays.

## 7. Is PERSUADE 2.0 suitable for our structural-validity study?

**Yes.** It is one of the few large-scale corpora linking holistic AES scores, prompts, tasks, discourse-element labels, character spans, and effectiveness ratings for argumentative student writing.

## 8. What limitations must the paper disclose?

- Long-format table: essay fields duplicated per segment
- `Unannotated` segments fill gaps; effectiveness only on rhetorical types
- Offset/`discourse_text` mismatches (~trimmed text); reconstruct from offsets
- Incomplete partition coverage for many essays (trailing unannotated chars)
- 2 essays with conflicting full text; 1 essay in both splits
- Grade levels 6–12 with gaps (no grade 7 in data)
- Task label is `Text dependent`, not `Source-based`
- Official test set includes holistic labels (not blind); fine for structure research, note for AES benchmarking

## 9. Proceed, proceed with exclusions, or reject?

### PROCEED WITH EXCLUSIONS

Evidence: 100.0% essays pass strict validation; all required fields present; 100.0% have text+score+annotations. Exclude essays with conflicting text, cross-split duplication, and optionally essays with severe partition/offset failures.

## Score audit

| score | essay_count | percentage |
| --- | --- | --- |
| 1.0 | 1028.0 | 3.9548 |
| 2.0 | 5699.0 | 21.9243 |
| 3.0 | 8368.0 | 32.192 |
| 4.0 | 6729.0 | 25.8867 |
| 5.0 | 3297.0 | 12.6837 |
| 6.0 | 873.0 | 3.3585 |

## Prompt inventory (counts)

| prompt_name | essay_count | task_types | unique_scores |
| --- | --- | --- | --- |
| Facial action coding system | 2167 | Text dependent | 1|2|3|4|5|6 |
| Distance learning | 2157 | Independent | 1|2|3|4|5|6 |
| Does the electoral college work? | 2046 | Text dependent | 1|2|3|4|5|6 |
| Car-free cities | 1959 | Text dependent | 1|2|3|4|5|6 |
| Driverless cars | 1886 | Text dependent | 1|2|3|4|5|6 |
| Exploring Venus | 1862 | Text dependent | 1|2|3|4|5|6 |
| Summer projects | 1750 | Independent | 1|2|3|4|5|6 |
| Mandatory extracurricular activities | 1670 | Independent | 1|2|3|4|5|6 |
| Cell phones at school | 1656 | Independent | 1|2|3|4|5|6 |
| Grades for extracurricular activities | 1626 | Independent | 1|2|3|4|5|6 |
| The Face on Mars | 1583 | Text dependent | 1|2|3|4|5|6 |
| Seeking multiple opinions | 1552 | Independent | 1|2|3|4|5|6 |
| Community service | 1542 | Independent | 1|2|3|4|5|6 |
| "A Cowboy Who Rode the Waves" | 1372 | Text dependent | 1|2|3|4|5 |
| Phones and driving | 1166 | Independent | 1|2|3|4|5|6 |

## Structure coverage (join-valid essays)

- Mean annotation coverage: 98.57%
- Mean components per essay: 9.15
- % with Claim: 95.6%
- % with Evidence: 99.7%
- % with Counterclaim: 29.3%
- % with Rebuttal: 23.2%
- % with Concluding Statement: 85.2%

## Official splits (Step 10)

- Split column: `competition_set`
- **test**: 10,401 essays; holistic scores present for 10,401 (100.0%)
- **train**: 15,592 essays; holistic scores present for 15,592 (100.0%)
- No separate validation split provided; only train/test
- Test holistic labels **are accessible** in released PERSUADE 2.0 files
- Official split suitable for discourse+structure research; document that test labels are not hidden

## Overall statistics by score (join-valid essays)

| holistic_essay_score | essays | mean_word_count | mean_components | pct_claim | pct_evidence |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 1028.0 | 279.6964980544747 | 3.708171206225681 | 0.4046692607003891 | 0.9785992217898832 |
| 2.0 | 5699.0 | 258.24249868397965 | 6.754167397789086 | 0.9252500438673451 | 0.9910510615897525 |
| 3.0 | 8368.0 | 351.0911806883365 | 8.95195984703633 | 0.9904397705544933 | 0.9995219885277247 |
| 4.0 | 6727.0 | 479.0612457261781 | 10.424409097666121 | 0.9976215251969675 | 1.0 |
| 5.0 | 3297.0 | 657.1355777979982 | 11.694570821959356 | 0.9963603275705186 | 1.0 |
| 6.0 | 873.0 | 843.7216494845361 | 13.780068728522338 | 0.9988545246277205 | 1.0 |
