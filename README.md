# ParsHate: A Benchmark Dataset for Hate and Target Detection in Persian

<p align="center">
  <b>The first decade-long (2013–2022) benchmark for Persian hate-speech and target detection</b><br>
  Zahra Bokaei · Walid Magdy · Bonnie Webber<br>
  School of Informatics, University of Edinburgh<br>
  <i>EMNLP 2026</i>
</p>

<p align="center">
  <a href="#-dataset-at-a-glance">Overview</a> •
  <a href="#-annotation-scheme">Annotation</a> •
  <a href="#-data-format">Data format</a> •
  <a href="#-benchmark-results">Results</a> •
  <a href="#-citation">Citation</a>
</p>

---



---

## 📌 Overview

**ParsHate** is a manually annotated dataset of **10,000 Persian tweets** sampled evenly across **ten years (2013–2022)**. Beyond binary hate detection, it supports:

- 🎯 **Multi-label target identification** across 7 structured categories with fine-grained subcategories
- 🕵️ **Explicit vs. implicit hate**, with the rhetorical **strategy** used for implicit hate
- 👁️ **Explicit vs. implicit targets** (is the attacked group named, or only invoked?)
- 🖍️ **Span-level rationales** marking the textual evidence for each hate label
- 📅 **Longitudinal evaluation** of temporal distribution shift

Data were sampled with a hybrid **random + score-stratified temporal** strategy to reduce keyword-driven bias while keeping natural label distributions.

## 📊 Dataset at a Glance

| Property | Value |
|---|---|
| Source | Twitter (1% daily stream archive) |
| Period | 2013–2022 |
| Size | 10,000 tweets (1,000 per year) |
| Hate | 31% |
| Implicit hate (of hate tweets) | 7% |
| Implicit targets (of hate tweets) | 35% |
| Avg. rationale span coverage | 18% of tweet characters |
| Annotators | 3 native Persian speakers per tweet |

### Comparison with existing Persian datasets

| Dataset | Source | Period | Size | % hate | Implicit hate | Target | Implicit target |
|---|---|---|---|---|:---:|:---:|:---:|
| Pars-OFF (Ataei et al., 2022) | Tweets | 2015–2020 | 10,000 | 30% (OFF) | ✗ | ✗ | ✗ |
| PHATE (Delbari et al., 2024) | Tweets | 2020–2023 | 7,000 | 23% | ✗ | ✓ | ✗ |
| Pars-HAO (Sheykhlan et al., 2023) | Tweets | unknown | 8,000 | 9% | ✗ | ✗ | ✗ |
| PHICAD (Davardoust et al., 2024) | Instagram | unknown | 30,000 | 41% (HOF) | ✗ | ✗ | ✗ |
| **ParsHate (ours)** | Tweets | **2013–2022** | 10,000 | 31% | ✓ | ✓ | ✓ |

### Target distribution (share of hate instances)

| Politics | Religion | Nationality | Occupation | Ethnicity | Gender | Other |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 76.45% | 6.42% | 4.90% | 3.45% | 2.29% | 1.94% | 4.55% |

39% of political hate instances involve implicit target references.

## 🏗️ Data Collection

1. **Source pool.** ~47M Persian tweets from a 1% daily Twitter sample (2013–2022).
2. **Filtering.** Removed retweets, duplicates, tweets under five tokens, and tweets containing only URLs, mentions, or hashtags → **29M tweets**.
3. **Scoring.** A fine-tuned Llama 3 Persian hate classifier (Bokaei et al., ACL 2025) assigned each tweet a hate probability. Scores were used **only for sampling**, never for labelling. None of the classifier's training data overlaps with ParsHate.
4. **Hybrid sampling (per year, 1,000 tweets):**
   - 500 tweets sampled **uniformly at random** (natural discourse)
   - 500 tweets **stratified by score**: 50 from each probability bin (0.0–0.1, …, 0.9–1.0)

## 🏷️ Annotation Scheme

Hate is defined following Waseem and Hovy (2016): content that expresses or encourages hostility toward groups or individuals based on their association with identity-related categories. Violent or merely offensive content is **not** hate unless it targets identity (see the guidelines).

| Layer | Labels |
|---|---|
| **Hate** | hate / non-hate |
| **Hate explicitness** | explicit / implicit |
| **Implicit strategy** (multi-label) | cultural/historical reference · sarcasm/figurative language · other indirect |
| **Target** (multi-label) | Gender (women/men) · Politics (domestic/foreign) · Religion (Islam/other) · Nationality (Iranian/Afghan/Arab/Western/other) · Occupation · Ethnicity · Other |
| **Target explicitness** | explicit / implicit |
| **Rationale spans** | character-level spans kept where ≥ 2 annotators agree |

### Inter-annotator agreement (Fleiss' κ)

| Task | κ |
|---|---|
| Hate identification | 0.69 |
| Target category | 0.60 |
| Implicit hate | 0.51 |
| Implicit target | 0.58 |
| Spans (overall / explicit / implicit) | 0.55 / 0.64 / 0.44 |

Disagreements were resolved by majority vote plus guideline-based adjudication. The full annotation guidelines (English translation) are in Appendix A.5 of the paper; the original Persian guidelines are included in this repository.

## 📁 Data Format

Each row corresponds to one tweet. All labels are binary (`1` = present, `0` = absent).

| Column | Description |
|---|---|
| `id` | Anonymised tweet ID |
| `year` | Year of posting (2013–2022) |
| `text` | Tweet text (user identifiers removed) |
| `hate` | 1 if hateful |
| `hate_implicit` | 1 if hate is implicit |
| `strategy_reference` / `strategy_figurative` / `strategy_other` | Implicit strategy flags |
| `target_*` | One column per target category (and subcategory) |
| `target_implicit` | 1 if the target is not directly named |
| `spans` | Rationale spans as character offsets |

### Loading

```python
import pandas as pd

df = pd.read_csv("data/parshate.csv")
hate = df[df["hate"] == 1]
print(f"{len(df)} tweets, {df['hate'].mean():.1%} hate")
```

## 🧪 Benchmark Results

All fine-tuned results are **macro-F1**, averaged over three seeds (mean ± std). Splits are a fixed per-year 80/20.

### Hate detection

| Setting | Model | Macro-F1 |
|---|---|:---:|
| Zero-shot | LLaMA-3 | 49.9 |
| Zero-shot | Gemma-3 | 53.2 |
| Zero-shot | GPT-5 | 68.0 |
| PHATE-trained SOTA | LLaMA-3 (Bokaei et al., 2025) | 74.8 |
| Year-specific fine-tuning | ParsBERT | 67.7 ± 0.7 |
| Year-specific fine-tuning | LLaMA-3 | 63.5 ± 0.7 |
| Year-specific fine-tuning | Gemma-3 | 64.3 ± 0.7 |
| **All-years fine-tuning** | **LLaMA-3** | **79.0 ± 0.5** |
| All-years fine-tuning | Gemma-3 | 78.2 ± 0.5 |

**Temporal shift:** the PHATE-trained model exceeds 88 F1 on 2020–2022 but drops to the mid-50s–mid-60s on 2013–2015. Early years also show the highest lexical diversity (type–token ratio), partly driven by named entities.

### Explicit vs. implicit hate (all-years models)

| Model | Explicit F1 | Implicit F1 |
|---|:---:|:---:|
| LLaMA-3-FT | 84.5 | 4.4 |
| Gemma-3-FT | 83.7 | 6.2 |

Per-strategy implicit F1 (LLaMA-3-FT): cultural/historical reference **2**, sarcasm/figurative **4**, other indirect **5**. Cultural/historical reference is both the hardest and the most frequent strategy (48% of implicit hate).

### Multi-label target prediction (macro average)

| Model | P | R | F1 | F1 (NE-masked) |
|---|:---:|:---:|:---:|:---:|
| GPT-5 | 25.5 | 56.6 | 23.8 | 24.4 |
| LLaMA-3-FT | 42.0 | 23.0 | 26.0 | 27.1 |

LLaMA-3 is strong on Politics (F1 = 90) but collapses on minority targets. GPT-5 achieves higher minority-class recall at a precision cost.

### Span detection (token-level F1)

| Model | Overall | Explicit | Implicit |
|---|:---:|:---:|:---:|
| LLaMA-3-FT | 44 | 58 | 14 |

## 🔑 Key Findings

- **Temporal diversity helps:** training on all years beats both a SOTA model trained on recent data and year-specific models.
- **Implicit hate is nearly unsolved:** models score ~84 F1 on explicit hate but under 7 on implicit hate.
- **Knowledge is the bottleneck:** failures concentrate on culturally and historically grounded references specific to Persian discourse.
- **Implicit targets are hard:** they make up 35% of hate instances but ~58% of target errors; minority targets are often misassigned to Politics.

## ⚖️ Ethics and Intended Use

- Built from a public archival Twitter sample; **user identifiers are removed**, and no attempt is made to identify individuals.
- Annotators were informed of distressing content, could take breaks or opt out at any time, and were compensated at a fixed rate.
- Released **for research only**. Please treat the data as sensitive, do not attempt to re-identify users, and do not use it to target or profile individuals or groups.

## 🚧 Limitations

- Twitter only; may not generalise to Instagram, Telegram, or other platforms.
- Strong class imbalance (politics dominates targets).
- Implicit hate is a small share of the data, limiting fine-grained modelling.

## 📖 Citation

If you use ParsHate, please cite:

```bibtex
@inproceedings{bokaei2026parshate,
  title     = {ParsHate: A Benchmark Dataset for Hate and Target Detection in Persian},
  author    = {Bokaei, Zahra and Magdy, Walid and Webber, Bonnie},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```
