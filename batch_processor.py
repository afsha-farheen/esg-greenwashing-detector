"""
ESG Greenwashing Detector — Batch Processor
=============================================
Reads every PDF in the 'reports/' folder, extracts text, splits it into
sentences, scores each sentence as Vague or Concrete, and tags it with
Sector / Company / Year (parsed from the filename).

Expected filename format:  Sector_Company_Year.pdf
    e.g. IT_Infosys_2025.pdf, Manufacturing_TataSteel_2024.pdf

Outputs (into 'output/' folder):
    1. esg_sentences_master.csv   -> every sentence, tagged + scored
    2. esg_summary_by_report.csv  -> specificity % per report
    3. esg_summary_by_sector.csv  -> specificity % per sector (for comparison)
    4. sector_comparison_chart.png -> bar chart of sector specificity scores

Run with:  python batch_processor.py
"""

import os
import re
import glob
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
REPORTS_FOLDER = "reports"
OUTPUT_FOLDER = "output"

VAGUE_WORDS = [
    "committed to", "strive", "striving", "believe in", "passionate about",
    "focus on", "focused on", "dedicated to", "aim to", "aspire",
    "greater consciousness", "core value", "at our core", "deeply care",
    "sustainable future", "responsible business", "making a difference",
    "our journey", "we care", "vision for", "leading the way"
]

# Signals that make a sentence "Concrete" / verifiable
PCT_PATTERN = re.compile(r"\d+(\.\d+)?\s?%")
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
QUANTITY_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s?(tonnes?|kg|mt|mw|gwh|kwh|crore|lakh|million|billion|litres?|km|acres?)\b",
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# STEP 1 — Parse filename into metadata
# ---------------------------------------------------------------------------
def parse_filename(filename):
    """
    Expects: Sector_Company_Year.pdf
    Falls back gracefully if the pattern doesn't fully match.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")

    sector = parts[0] if len(parts) > 0 else "Unknown"
    company = parts[1] if len(parts) > 1 else base
    year = parts[2] if len(parts) > 2 else "Unknown"

    return sector, company, year


# ---------------------------------------------------------------------------
# STEP 2 — Extract text from PDF
# ---------------------------------------------------------------------------
def detect_column_gap(page, bins=100):
    """
    Detects whether a PDF page uses a genuine two-column layout by
    looking for a vertical gap (no words) roughly in the middle of
    the page, with real text content on BOTH sides across several
    distinct lines. This avoids false positives from short
    single-column lines that just happen to end early (trailing
    whitespace is not the same as a real column gutter).

    Returns (gap_x0, gap_x1) if a genuine column gutter is found,
    otherwise None.
    """
    words = page.extract_words()
    if not words:
        return None

    page_width = page.width
    bin_width = page_width / bins
    covered = [False] * bins
    for w in words:
        start_bin = max(0, int(w["x0"] / bin_width))
        end_bin = min(bins - 1, int(w["x1"] / bin_width))
        for b in range(start_bin, end_bin + 1):
            covered[b] = True

    # Look for the widest uncovered run within the middle 30%-70% of
    # the page (a real column gutter is never at the very edges).
    mid_start, mid_end = int(bins * 0.3), int(bins * 0.7)
    best_gap = (0, 0)
    run_start = None
    for i in range(mid_start, mid_end + 1):
        if not covered[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) > (best_gap[1] - best_gap[0]):
                best_gap = (run_start, i)
            run_start = None
    if run_start is not None and (mid_end + 1 - run_start) > (best_gap[1] - best_gap[0]):
        best_gap = (run_start, mid_end + 1)

    gap_width = best_gap[1] - best_gap[0]
    if gap_width < 2:  # lowered from 3 -- narrow column gutters (as
        return None    # seen in portrait-page reports) were being missed

    gap_x0 = best_gap[0] * bin_width
    gap_x1 = best_gap[1] * bin_width

    # Require real content on BOTH sides of the gap across several
    # distinct lines -- confirms this is an actual two-column layout,
    # not just short lines leaving blank space at the page's right edge.
    line_groups = {}
    for w in words:
        line_key = round(w["top"] / 3)
        line_groups.setdefault(line_key, []).append(w)

    lines_with_both_sides = 0
    for line_words in line_groups.values():
        has_left = any(w["x1"] <= gap_x0 for w in line_words)
        has_right = any(w["x0"] >= gap_x1 for w in line_words)
        if has_left and has_right:
            lines_with_both_sides += 1

    if lines_with_both_sides < 3:
        return None

    return gap_x0, gap_x1


def detect_repeated_lines(pdf, min_pages=4, freq_threshold=0.25):
    """
    Finds lines (like page headers/footers) that repeat identically
    across many pages of the SAME pdf -- these are boilerplate, not
    real ESG content, and should be stripped before sentence splitting.
    Only applies the filter for documents with enough pages to make
    the pattern meaningful (avoids wrongly stripping real content
    from short reports).
    """
    if len(pdf.pages) < min_pages:
        return set()

    line_counts = {}
    for page in pdf.pages:
        raw = page.extract_text() or ""
        seen_this_page = set()
        for line in raw.split("\n"):
            line = line.strip()
            if len(line) < 8:  # too short to be a meaningful header/footer
                continue
            if line not in seen_this_page:
                line_counts[line] = line_counts.get(line, 0) + 1
                seen_this_page.add(line)

    total_pages = len(pdf.pages)
    repeated = {
        line for line, count in line_counts.items()
        if count / total_pages >= freq_threshold
    }
    return repeated


def strip_repeated_lines(text, repeated_lines):
    if not repeated_lines:
        return text
    kept = [line for line in text.split("\n") if line.strip() not in repeated_lines]
    return "\n".join(kept)


def extract_text_from_pdf(path):
    text_chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            repeated_lines = detect_repeated_lines(pdf)

            for page in pdf.pages:
                try:
                    gap = detect_column_gap(page)
                except Exception:
                    gap = None

                if gap:
                    gap_x0, gap_x1 = gap
                    left = page.crop((0, 0, gap_x0, page.height))
                    right = page.crop((gap_x1, 0, page.width, page.height))
                    left_text = left.extract_text() or ""
                    right_text = right.extract_text() or ""
                    page_text = left_text + "\n" + right_text
                else:
                    page_text = page.extract_text(layout=True) or page.extract_text()

                if page_text:
                    page_text = strip_repeated_lines(page_text, repeated_lines)
                    text_chunks.append(page_text)
    except Exception as e:
        print(f"  [!] Could not read {path}: {e}")
    return "\n".join(text_chunks)


# ---------------------------------------------------------------------------
# STEP 3 — Clean + split into sentences
# ---------------------------------------------------------------------------
def fix_encoding_glitches(text):
    """
    Fixes common 'mojibake' artifacts from PDF text extraction, where
    curly quotes/apostrophes get mangled into sequences like â€™.
    """
    replacements = {
        "â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"',
        "â€“": "-", "â€”": "-", "â€¦": "...", "Â": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def looks_scrambled(sentence):
    """
    Detects sentences that are likely garbled column-merges from
    multi-column PDF layouts (e.g., financial tables merged with
    narrative text). Heuristic: too many separate numeric/currency
    tokens crammed into one 'sentence' relative to its word count.
    """
    words = sentence.split()
    if len(words) == 0:
        return True

    numeric_tokens = re.findall(
        r"(\$\s?\d|\d+(\.\d+)?\s?(/t|mt|%|t\b)|\bFY\d{2,4}\b)",
        sentence, re.IGNORECASE
    )
    # If more than ~15% of words are numeric/financial tokens, it's
    # very likely a scrambled table/column merge, not a real sentence.
    if len(numeric_tokens) / len(words) > 0.15 and len(numeric_tokens) >= 3:
        return True

    # Sudden case-shift + orphan capital fragments mid-sentence is
    # another sign of column-jumbling (e.g., ". The Company continues")
    abrupt_caps = re.findall(r"[a-z]\.\s+[A-Z][a-z]+\s[A-Z][a-z]+.{0,20}[a-z]\.\s+[A-Z]", sentence)
    if len(abrupt_caps) >= 2:
        return True

    return False


ESG_KEYWORDS = [
    # Environmental
    "emission", "carbon", "climate", "renewable", "energy", "water",
    "waste", "recycl", "biodiversity", "pollution", "environment",
    "sustainab", "green", "net zero", "netzero", "decarbon", "ghg",
    "circular economy", "afforestation", "solar", "wind power",
    # Social
    "employee", "diversity", "inclusion", "safety", "human rights",
    "labour", "labor", "community", "csr", "corporate social",
    "wellbeing", "well-being", "training", "gender", "workforce",
    "supply chain", "child labour", "health and safety", "philanthropy",
    "livelihood",
    # Governance (ESG-specific, not general corporate/financial)
    "governance", "ethics", "anti-corruption", "whistle-blower",
    "whistleblower", "transparency", "compliance", "board diversity",
    "stakeholder", "esg", "brsr", "materiality",
]


def is_esg_relevant(sentence):
    """
    Checks whether a sentence is actually about an ESG topic at all.
    Full annual reports contain huge amounts of financial statements,
    legal/AGM procedure, and board-compensation tables that happen to
    contain numbers -- these should NOT be scored as ESG claims just
    because they pass the Concrete/Vague number check.
    """
    lower = sentence.lower()
    return any(keyword in lower for keyword in ESG_KEYWORDS)


def split_into_sentences(text):
    text = fix_encoding_glitches(text)
    # Collapse whitespace/newlines from PDF extraction artifacts
    text = re.sub(r"\s+", " ", text)
    # Basic sentence split on . ! ? followed by a space and an uppercase
    # letter OR a digit (so "...12 factories achieved..." also splits correctly)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    # Drop junk: too short, mostly numbers/symbols, page-number noise
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) < 5:
            continue
        if len(s) > 500:  # runaway match, likely bad split
            continue
        if looks_scrambled(s):
            continue  # skip likely column-merge garbage from PDF layout
        if not is_esg_relevant(s):
            continue  # skip non-ESG content (financials, legal/AGM text, etc.)
        cleaned.append(s)
    return cleaned


# ---------------------------------------------------------------------------
# STEP 4 — Score a sentence: Vague vs Concrete
# ---------------------------------------------------------------------------
def score_sentence(sentence):
    score = 0
    reasons = []

    if PCT_PATTERN.search(sentence):
        score += 1
        reasons.append("contains %")
    if YEAR_PATTERN.search(sentence):
        score += 1
        reasons.append("contains year")
    if QUANTITY_PATTERN.search(sentence):
        score += 1
        reasons.append("contains measurable quantity")

    lower = sentence.lower()
    vague_hit = any(vw in lower for vw in VAGUE_WORDS)

    if score >= 1:
        verdict = "Concrete"
    elif vague_hit:
        verdict = "Vague"
        reasons.append("vague trigger phrase")
    else:
        verdict = "Neutral"  # no hard evidence, but no vague trigger either
        reasons.append("no strong signal either way")

    return verdict, score, "; ".join(reasons)


# ---------------------------------------------------------------------------
# STEP 5 — Process every PDF in the reports folder
# ---------------------------------------------------------------------------
def process_all_reports():
    pdf_paths = glob.glob(os.path.join(REPORTS_FOLDER, "*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in '{REPORTS_FOLDER}/'. "
              f"Make sure your reports are saved there.")
        return None

    all_rows = []
    print(f"Found {len(pdf_paths)} PDF(s). Processing...\n")

    for path in pdf_paths:
        sector, company, year = parse_filename(path)
        print(f"  -> {os.path.basename(path)}  [{sector} | {company} | {year}]")

        raw_text = extract_text_from_pdf(path)
        if not raw_text.strip():
            print(f"     [!] No extractable text found — skipping.")
            continue

        sentences = split_into_sentences(raw_text)
        for sent in sentences:
            verdict, score, reason = score_sentence(sent)
            all_rows.append({
                "sector": sector,
                "company": company,
                "year": year,
                "source_file": os.path.basename(path),
                "sentence": sent,
                "verdict": verdict,
                "score": score,
                "reason": reason
            })

    df = pd.DataFrame(all_rows)
    print(f"\nTotal sentences extracted across all reports: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# STEP 6 — Summaries
# ---------------------------------------------------------------------------
def build_summaries(df):
    def specificity_pct(sub_df):
        total = len(sub_df)
        concrete = (sub_df["verdict"] == "Concrete").sum()
        return round((concrete / total) * 100, 1) if total > 0 else 0

    # By report (company + year)
    report_rows = []
    for (company, year, sector), group in df.groupby(["company", "year", "sector"]):
        report_rows.append({
            "sector": sector,
            "company": company,
            "year": year,
            "total_sentences": len(group),
            "concrete_count": (group["verdict"] == "Concrete").sum(),
            "vague_count": (group["verdict"] == "Vague").sum(),
            "neutral_count": (group["verdict"] == "Neutral").sum(),
            "specificity_pct": specificity_pct(group)
        })
    report_summary = pd.DataFrame(report_rows).sort_values("specificity_pct", ascending=False)

    # By sector (the key input for your comparison feature)
    sector_rows = []
    for sector, group in df.groupby("sector"):
        sector_rows.append({
            "sector": sector,
            "total_sentences": len(group),
            "concrete_count": (group["verdict"] == "Concrete").sum(),
            "vague_count": (group["verdict"] == "Vague").sum(),
            "specificity_pct": specificity_pct(group)
        })
    sector_summary = pd.DataFrame(sector_rows).sort_values("specificity_pct", ascending=False)

    return report_summary, sector_summary


# ---------------------------------------------------------------------------
# STEP 7 — Chart
# ---------------------------------------------------------------------------
def plot_sector_comparison(sector_summary, out_path):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(sector_summary["sector"], sector_summary["specificity_pct"],
                    color="#4C72B0")
    plt.ylabel("Specificity Score (%)")
    plt.title("ESG Report Specificity by Sector\n(Higher = more verifiable, less vague)")
    plt.ylim(0, 100)
    for bar, pct in zip(bars, sector_summary["specificity_pct"]):
        plt.text(bar.get_x() + bar.get_width() / 2, pct + 1, f"{pct}%",
                  ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    df = process_all_reports()
    if df is None or df.empty:
        return

    master_path = os.path.join(OUTPUT_FOLDER, "esg_sentences_master.csv")
    df.to_csv(master_path, index=False)
    print(f"\nSaved master sentence-level file -> {master_path}")

    report_summary, sector_summary = build_summaries(df)

    report_path = os.path.join(OUTPUT_FOLDER, "esg_summary_by_report.csv")
    sector_path = os.path.join(OUTPUT_FOLDER, "esg_summary_by_sector.csv")
    report_summary.to_csv(report_path, index=False)
    sector_summary.to_csv(sector_path, index=False)
    print(f"Saved per-report summary   -> {report_path}")
    print(f"Saved per-sector summary   -> {sector_path}")

    chart_path = os.path.join(OUTPUT_FOLDER, "sector_comparison_chart.png")
    plot_sector_comparison(sector_summary, chart_path)
    print(f"Saved comparison chart     -> {chart_path}")

    print("\n--- Sector Specificity Summary ---")
    print(sector_summary.to_string(index=False))

    print("\nDone. Open 'output/esg_sentences_master.csv' to start labeling "
          "sentences for your ML classifier (v2).")


if __name__ == "__main__":
    main()
