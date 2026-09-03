"""
ESG Greenwashing Detector — ML Classifier Training (v2)
=========================================================
Trains a Logistic Regression classifier on TF-IDF features to predict
Vague vs Concrete, using your manually labeled sentences
(output/manual_labels.csv).

Handles class imbalance (in this dataset, Concrete sentences greatly
outnumber Vague ones) using class_weight='balanced', and evaluates
using precision/recall/F1 -- NOT just accuracy, since accuracy alone
would be misleading on an imbalanced dataset.

Also compares performance against the rule-based v1 scorer on the
same held-out test sentences, where a rule-based verdict is available.

Outputs:
    - Prints evaluation metrics + confusion matrix to the terminal
    - Saves the trained model + vectorizer to output/model/ for reuse
    - Saves a labeled_predictions.csv showing test-set predictions

Run with:  python train_classifier.py
"""

import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

LABELS_FILE = os.path.join("output", "manual_labels.csv")
MASTER_FILE = os.path.join("output", "esg_sentences_master.csv")
MODEL_DIR = os.path.join("output", "model")


def load_data():
    if not os.path.exists(LABELS_FILE):
        print(f"Could not find {LABELS_FILE}. Run label_sentences.py first.")
        return None
    df = pd.read_csv(LABELS_FILE)
    df = df.dropna(subset=["sentence", "manual_label"])
    return df


def train_and_evaluate(df):
    X = df["sentence"]
    y = df["manual_label"]

    # Stratified split keeps the same Vague/Concrete ratio in both
    # train and test sets, despite the class imbalance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} sentences, testing on {len(X_test)}.")
    print(f"Train label counts:\n{y_train.value_counts()}\n")
    print(f"Test label counts:\n{y_test.value_counts()}\n")

    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),   # unigrams + bigrams (captures phrases like "committed to")
        stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",  # compensates for the Concrete/Vague imbalance
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("=" * 60)
    print("ML CLASSIFIER (v2) EVALUATION")
    print("=" * 60)
    print(f"\nOverall accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nDetailed report (precision/recall/F1 per class):")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"           Predicted:  {'  '.join(labels)}")
    for i, row_label in enumerate(labels):
        print(f"Actual {row_label:10s}   {cm[i]}")

    return model, vectorizer, X_test, y_test, y_pred


def save_model(model, vectorizer):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "classifier.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\nModel saved to {MODEL_DIR}/")


def compare_with_rule_based(X_test, y_test, y_pred):
    """
    Compares the ML model's predictions against the original rule-based
    v1 verdicts, for the same test sentences, where a v1 verdict exists
    (v1 also has a 'Neutral' category which isn't directly comparable,
    so those rows are excluded from this specific comparison).
    """
    if not os.path.exists(MASTER_FILE):
        print("\n(Skipping v1 comparison -- esg_sentences_master.csv not found)")
        return

    master = pd.read_csv(MASTER_FILE)[["sentence", "verdict"]].drop_duplicates(subset="sentence")

    test_df = pd.DataFrame({
        "sentence": X_test.values,
        "manual_label": y_test.values,
        "ml_prediction": y_pred
    })
    merged = test_df.merge(master, on="sentence", how="left")
    merged = merged[merged["verdict"].isin(["Concrete", "Vague"])]

    if merged.empty:
        print("\n(No overlapping rule-based verdicts found for comparison)")
        return

    v1_correct = (merged["verdict"] == merged["manual_label"]).mean()
    v2_correct = (merged["ml_prediction"] == merged["manual_label"]).mean()

    print("\n" + "=" * 60)
    print("RULE-BASED v1  vs  ML CLASSIFIER v2")
    print("=" * 60)
    print(f"\n--- Biased comparison (only sentences where v1 dared to")
    print(f"    call Concrete/Vague, skipping v1's 'Neutral' calls) ---")
    print(f"Sentences compared: {len(merged)}")
    print(f"v1 accuracy on this easy subset: {v1_correct:.2%}")
    print(f"v2 accuracy on this same subset: {v2_correct:.2%}")
    print(f"(This favors v1 -- it only had to judge the cases its own")
    print(f" rules were confident about. Not a fair overall comparison.)")

    # FAIR comparison: score v1 on the FULL test set, treating any
    # non-Concrete/Vague verdict (i.e. "Neutral") as v1 failing to
    # correctly identify the sentence -- since v1 was asked to decide
    # on every sentence, not just the easy ones.
    full_test_df = pd.DataFrame({
        "sentence": X_test.values,
        "manual_label": y_test.values,
    }).merge(master, on="sentence", how="left")

    full_test_df["v1_correct"] = full_test_df["verdict"] == full_test_df["manual_label"]
    v1_full_accuracy = full_test_df["v1_correct"].mean()
    v2_full_accuracy = (y_pred == y_test.values).mean()

    print(f"\n--- Fair comparison (ALL {len(full_test_df)} test sentences,")
    print(f"    v1's 'Neutral'/missing calls counted as incorrect) ---")
    print(f"v1 (rule-based) accuracy: {v1_full_accuracy:.2%}")
    print(f"v2 (ML model)   accuracy: {v2_full_accuracy:.2%}")


def main():
    df = load_data()
    if df is None or df.empty:
        return

    model, vectorizer, X_test, y_test, y_pred = train_and_evaluate(df)
    save_model(model, vectorizer)
    compare_with_rule_based(X_test, y_test, y_pred)

    # Save test predictions for inspection
    out_path = os.path.join("output", "test_predictions.csv")
    pd.DataFrame({
        "sentence": X_test.values,
        "true_label": y_test.values,
        "predicted_label": y_pred
    }).to_csv(out_path, index=False)
    print(f"\nTest set predictions saved to {out_path}")


if __name__ == "__main__":
    main()
