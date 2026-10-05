"""
train_model.py
----------------
Generates a labelled training dataset and trains a Random Forest classifier
to detect phishing URLs from the lexical/host-based feature set defined in
features.py.

DATA NOTE
---------
This project ships with a *synthetically generated* dataset rather than a
downloaded one, so that the whole pipeline (data -> features -> model) runs
fully offline and reproducibly in any environment, including sandboxes with
restricted internet access. The generator encodes the same statistical
patterns documented in phishing-detection literature (long/obfuscated URLs,
IP-address hosts, suspicious keywords, link shorteners, risky TLDs, etc.)
with randomised noise, producing a dataset that is realistic enough to train
and meaningfully evaluate a classifier.

To use REAL data instead (recommended for production), simply:
  1. Obtain a labelled dataset such as the UCI "Phishing Websites" dataset
     or a PhishTank / OpenPhish export of confirmed phishing URLs plus a
     legitimate-URL sample (e.g., from a top-sites list).
  2. Run every URL through `features.extract_all_features()` to build a
     feature matrix with the same column order (FEATURE_ORDER).
  3. Point `train_from_dataframe()` below at that DataFrame instead of
     `generate_synthetic_dataset()`.

Run:
    python train_model.py
"""

import json
import random
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import FEATURE_ORDER

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

LEGIT_BRANDS = ["google", "amazon", "microsoft", "github", "wikipedia", "apple",
                "netflix", "linkedin", "paypal", "yahoo", "spotify", "reddit",
                "dropbox", "adobe", "salesforce", "cloudflare"]
LEGIT_TLDS = ["com", "org", "net", "edu", "io", "co"]
SUSPICIOUS_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "loan"]
SUSPICIOUS_WORDS = ["login", "verify", "secure", "account", "update", "confirm",
                     "bank", "signin", "password", "billing", "suspend", "recover",
                     "unlock", "wallet", "alert"]
SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "cutt.ly"]


def _rand_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _rand_token(n=6):
    letters = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(letters) for _ in range(n))


def _make_legit_url():
    brand = random.choice(LEGIT_BRANDS)
    tld = random.choice(LEGIT_TLDS)
    sub = random.choice(["", "www.", "docs.", "mail.", "support."])
    path = random.choice(["", "/", "/home", "/products", f"/{_rand_token(5)}", "/help/article"])
    return f"https://{sub}{brand}.{tld}{path}"


def _make_phishing_url():
    style = random.choice(["ip_host", "shortener", "suspicious_word", "fake_subdomain",
                            "suspicious_tld", "at_symbol", "long_obfuscated"])
    brand = random.choice(LEGIT_BRANDS)
    word = random.choice(SUSPICIOUS_WORDS)

    if style == "ip_host":
        return f"http://{_rand_ip()}/{word}/{_rand_token(8)}"
    if style == "shortener":
        return f"http://{random.choice(SHORTENERS)}/{_rand_token(7)}"
    if style == "suspicious_word":
        tld = random.choice(LEGIT_TLDS + SUSPICIOUS_TLDS)
        return f"http://{word}-{brand}-{_rand_token(4)}.{tld}/{word}"
    if style == "fake_subdomain":
        tld = random.choice(SUSPICIOUS_TLDS + LEGIT_TLDS)
        return f"http://{brand}.{word}.{_rand_token(5)}.{tld}/"
    if style == "suspicious_tld":
        tld = random.choice(SUSPICIOUS_TLDS)
        return f"http://{brand}-secure.{tld}/{word}.php"
    if style == "at_symbol":
        tld = random.choice(LEGIT_TLDS)
        return f"http://{brand}.{tld}@{_rand_ip()}/{word}"
    # long_obfuscated
    tld = random.choice(SUSPICIOUS_TLDS + LEGIT_TLDS)
    junk = "-".join(_rand_token(4) for _ in range(4))
    return f"http://{brand}-{word}-{junk}.{tld}/{_rand_token(10)}/{_rand_token(10)}"


def _make_hard_legit_url():
    """A legitimate-style URL that nonetheless trips one or two surface
    heuristics (long marketing/tracking paths, a brand name combined with a
    campaign word, etc.) -- included so the classifier has to learn more
    than a single shallow rule and so evaluation accuracy is realistic."""
    brand = random.choice(LEGIT_BRANDS)
    tld = random.choice(LEGIT_TLDS)
    word = random.choice(["update", "account", "secure", "login", "confirm"])
    junk = _rand_token(random.randint(6, 10))
    style = random.choice(["long_path", "word_subdomain", "tracking_query"])
    if style == "long_path":
        return f"https://www.{brand}.{tld}/{word}/{junk}/{_rand_token(6)}"
    if style == "word_subdomain":
        return f"https://{word}.{brand}.{tld}/"
    return f"https://www.{brand}.{tld}/campaign?ref={junk}&id={random.randint(1000,99999)}"


def _make_hard_phishing_url():
    """A phishing-style URL that mimics several legitimate surface traits
    (clean-looking domain, HTTPS, no obvious suspicious keyword) to model
    more sophisticated, low-signal phishing attempts."""
    tld = random.choice(LEGIT_TLDS)
    junk = _rand_token(random.randint(5, 9))
    style = random.choice(["clean_https", "lookalike_domain", "short_path"])
    if style == "clean_https":
        return f"https://{junk}-{_rand_token(4)}.{tld}/"
    if style == "lookalike_domain":
        brand = random.choice(LEGIT_BRANDS)
        swapped = brand.replace("o", "0").replace("l", "1") if any(c in brand for c in "ol") else brand + "s"
        return f"https://{swapped}.{tld}/"
    return f"https://{junk}.{tld}/{_rand_token(3)}"


def generate_synthetic_dataset(n_per_class=1500, hard_fraction=0.22):
    """Builds a balanced synthetic dataset of (url, label) pairs, label 1=phishing, 0=legitimate.
    A configurable fraction of each class is drawn from 'hard' generators that
    overlap in feature space with the opposite class, so the resulting dataset
    is not trivially/linearly separable and evaluation metrics are realistic."""
    rows = []
    n_hard = int(n_per_class * hard_fraction)
    n_easy = n_per_class - n_hard

    for _ in range(n_easy):
        rows.append((_make_legit_url(), 0))
    for _ in range(n_hard):
        rows.append((_make_hard_legit_url(), 0))

    for _ in range(n_easy):
        rows.append((_make_phishing_url(), 1))
    for _ in range(n_hard):
        rows.append((_make_hard_phishing_url(), 1))

    random.shuffle(rows)
    df = pd.DataFrame(rows, columns=["url", "label"])

    # Inject a small amount of realistic label noise (~2%) to mimic
    # ambiguous/mislabelled real-world samples and avoid a perfect-separation
    # toy problem.
    noise_idx = df.sample(frac=0.02, random_state=RANDOM_SEED).index
    df.loc[noise_idx, "label"] = 1 - df.loc[noise_idx, "label"]

    return df


def build_feature_matrix(df):
    """Runs every URL through the *offline* lexical feature extractor
    (host/DNS/TLS checks are skipped during training since large-scale
    live network probing of thousands of synthetic hosts is neither
    meaningful nor reliable in a training context)."""
    from features import extract_lexical_features

    records = []
    for url in df["url"]:
        feats, _, _ = extract_lexical_features(url)
        # Training uses lexical features only; dns_resolves / has_valid_tls
        # default to 0 here and are added live at inference time.
        feats["dns_resolves"] = 0
        feats["has_valid_tls"] = 0
        records.append(feats)
    X = pd.DataFrame(records)[FEATURE_ORDER]
    return X


def train_and_evaluate():
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset(n_per_class=1500)
    X = build_feature_matrix(df)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": len(FEATURE_ORDER),
    }
    print("Metrics:", json.dumps(metrics, indent=2))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    # Save model
    with open(MODEL_DIR / "phishing_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(FEATURE_ORDER, f, indent=2)

    # ---------------- Confusion Matrix Plot ----------------
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Legitimate", "Phishing"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14, weight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # ---------------- ROC Curve Plot ----------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#2563EB", linewidth=2.2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "roc_curve.png", dpi=160)
    plt.close()

    # ---------------- Feature Importance Plot ----------------
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:12]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([FEATURE_ORDER[i] for i in idx][::-1], importances[idx][::-1], color="#1E2761")
    ax.set_xlabel("Importance")
    ax.set_title("Top 12 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importance.png", dpi=160)
    plt.close()

    print("\nModel, metrics, and plots saved to:", MODEL_DIR)
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
