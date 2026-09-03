"""
ESG Greenwashing Detector — Interactive Sentence Labeler
==========================================================
Lets you manually label sentences from esg_sentences_master.csv as
Vague or Concrete, to build a training dataset for the ML classifier
(v2). Labels are saved incrementally, so you can stop anytime and
resume later without losing progress or re-labeling the same sentences.

Run with:  python label_sentences.py
"""

import os
import pandas as pd

INPUT_FILE = os.path.join("output", "esg_sentences_master.csv")
LABELS_FILE = os.path.join("output", "manual_labels.csv")
TARGET_COUNT = 250  # aim for ~250 labeled sentences


def load_sentences():
    df = pd.read_csv(INPUT_FILE)
    return df


def load_existing_labels():
    if os.path.exists(LABELS_FILE):
        return pd.read_csv(LABELS_FILE)
    return pd.DataFrame(columns=["sentence", "sector", "company", "manual_label"])


def save_label(sentence, sector, company, label):
    file_exists = os.path.exists(LABELS_FILE)
    row = pd.DataFrame([{
        "sentence": sentence,
        "sector": sector,
        "company": company,
        "manual_label": label
    }])
    row.to_csv(LABELS_FILE, mode="a", header=not file_exists, index=False)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Could not find {INPUT_FILE}. Run batch_processor.py first.")
        return

    df = load_sentences()
    existing = load_existing_labels()
    already_labeled = set(existing["sentence"].tolist()) if not existing.empty else set()

    remaining_needed = TARGET_COUNT - len(already_labeled)
    if remaining_needed <= 0:
        print(f"You've already labeled {len(already_labeled)} sentences -- "
              f"that meets your target of {TARGET_COUNT}!")
        print(f"Your labels are saved in: {LABELS_FILE}")
        return

    # Shuffle so you get a representative mix of sectors/verdicts,
    # not just whatever order they appear in the CSV
    pool = df[~df["sentence"].isin(already_labeled)].copy()
    pool = pool.sample(frac=1, random_state=None).reset_index(drop=True)

    print("=" * 70)
    print("ESG SENTENCE LABELER")
    print("=" * 70)
    print(f"Already labeled: {len(already_labeled)} / {TARGET_COUNT}")
    print(f"Need {remaining_needed} more.")
    print()
    print("For each sentence, decide: is this VAGUE (promotional, no hard")
    print("evidence) or CONCRETE (specific, verifiable claim)?")
    print()
    print("Type:  v = Vague   c = Concrete   s = Skip (unsure)   q = Quit & save")
    print("=" * 70)
    print()

    labeled_this_session = 0
    for _, row in pool.iterrows():
        if labeled_this_session >= remaining_needed:
            break

        print(f"\n[{row['sector']} | {row['company']} | {row['year']}]")
        print(f'"{row["sentence"]}"')
        print(f"(auto-scorer said: {row['verdict']})")

        choice = input("Your label (v/c/s/q): ").strip().lower()

        if choice == "q":
            print(f"\nStopped. You labeled {labeled_this_session} sentences this session.")
            print(f"Total labeled so far: {len(already_labeled) + labeled_this_session}")
            print(f"Resume anytime by running this script again.")
            return

        if choice not in ("v", "c", "s"):
            print("  (Not a valid input -- skipping this one.)")
            continue

        if choice == "s":
            continue  # don't save skipped sentences, just move on

        label = "Vague" if choice == "v" else "Concrete"
        save_label(row["sentence"], row["sector"], row["company"], label)
        labeled_this_session += 1
        print(f"  Saved as: {label}  "
              f"({len(already_labeled) + labeled_this_session}/{TARGET_COUNT} total)")

    total_labeled = len(already_labeled) + labeled_this_session
    if total_labeled >= TARGET_COUNT:
        print(f"\nTarget reached! You've labeled {TARGET_COUNT} sentences.")
    else:
        print(f"\nNo more unlabeled sentences left in the dataset.")
        print(f"You've labeled {total_labeled} total (target was {TARGET_COUNT}).")
    print(f"Your labels are saved in: {LABELS_FILE}")
    print("Next step: train the ML classifier on this labeled data.")


if __name__ == "__main__":
    main()
