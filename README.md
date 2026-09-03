# 🌱 ESG Greenwashing Detector

An NLP-based tool that detects **greenwashing** in corporate ESG (Environmental,
Social, Governance) reports — flagging vague, promotional sustainability
language versus specific, verifiable claims backed by real data.

**🔗 Live demo:** https://esg-greenwashing-detector-ltvri2yhwvkykvbawmbkl9.streamlit.app/
---

## The Problem

Companies publish ESG/sustainability reports every year, but much of the
language is vague and unverifiable ("committed to a greener future") rather
than specific and checkable ("reduced emissions by 18% in 2024"). Big 4 audit
firms and regulators increasingly need tools to flag this gap at scale. This
project builds a lightweight version of that kind of tool.

## What It Does

- Extracts and cleans text from real corporate ESG report PDFs
- Classifies each sentence as **Concrete** (specific, verifiable) or
  **Vague** (promotional, unverifiable) using two approaches:
  - **v1 — Rule-based:** regex + keyword scoring
  - **v2 — ML Classifier:** TF-IDF + Logistic Regression, trained on 250
    manually labeled sentences
- Compares **specificity scores across 4 industry sectors** (IT, Banking,
  Manufacturing, FMCG) using 9 real public sustainability reports
- Provides a live, interactive dashboard to analyze any new ESG report PDF

## Key Findings

- Across all 4 sectors analyzed, **no sector's ESG language crossed 30%
  specificity** — meaning the majority of sustainability claims, even from
  large listed companies, are promotional language without hard, checkable
  evidence behind them.
- FMCG (ITC) was the most specific sector (~27%); Banking (HDFC Life) was
  the most vague (~17%).
- Error analysis of the ML classifier's mistakes traced 100% of them back to
  residual PDF text-extraction noise (multi-column layout merging), not
  genuine weaknesses in distinguishing vague vs. concrete language — a
  useful diagnostic insight into where the true bottleneck was in the
  pipeline (data quality, not modeling approach).

## Tech Stack

- **Python** — pandas, pdfplumber, scikit-learn, matplotlib
- **NLP** — regex-based rule scoring, TF-IDF vectorization
- **ML** — Logistic Regression (with class-imbalance handling)
- **Dashboard** — Streamlit

## Project Structure

```
esg-greenwashing-detector/
├── batch_processor.py      # PDF extraction + rule-based scoring pipeline
├── label_sentences.py      # Interactive tool for manual sentence labeling
├── train_classifier.py     # Trains and evaluates the ML classifier (v2)
├── dashboard.py             # Streamlit dashboard (live demo)
├── requirements.txt
├── reports/                 # Source ESG report PDFs (not included in repo)
└── output/
    ├── esg_sentences_master.csv
    ├── esg_summary_by_sector.csv
    ├── esg_summary_by_report.csv
    ├── manual_labels.csv
    ├── sector_comparison_chart.png
    └── model/
        ├── classifier.pkl
        └── vectorizer.pkl
```

## Running It Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add ESG report PDFs to a 'reports/' folder,
#    named like: Sector_Company_Year.pdf (e.g. IT_Infosys_2025.pdf)

# 3. Run the extraction + scoring pipeline
python batch_processor.py

# 4. (Optional) Label sentences and train the ML classifier
python label_sentences.py
python train_classifier.py

# 5. Launch the dashboard
streamlit run dashboard.py
```

## Data Sources

Reports analyzed were sourced from public investor relations pages of:
TCS, Infosys, Tata Steel, HDFC Life, and ITC.

## Known Limitations

- Rule-based scoring can miss non-numeric but genuinely specific claims
  (e.g., named certifications without a percentage attached)
- PDF text extraction, while significantly improved through multi-column
  layout detection and header/footer stripping, can still produce occasional
  garbled sentences on complex magazine-style report layouts
- The ML classifier was trained on a relatively small (250-sentence) and
  imbalanced labeled dataset; a larger labeled set would likely improve
  its performance further
- This tool measures *specificity/verifiability* of language, not factual
  accuracy — it does not fact-check whether cited numbers are true

## Team

"This started as a college mini-project with a team of 4 — together we built the data pipeline and rule-based v1 detector. I then took it further on my own: I built the ML classifier (v2), did the error analysis comparing the two approaches, built the interactive dashboard, and deployed the whole thing publicly."

---

*Built as an academic mini-project exploring soft computing / AIML
applications in ESG auditing.*
