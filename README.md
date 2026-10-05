# PhishGuard — Phishing Detection System

A Flask + scikit-learn web application that analyses a URL and classifies it
as **Phishing** or **Legitimate** using a Random Forest model trained on a
24-dimension lexical and host-based feature vector.

## Features

- Single-URL analysis with a confidence score and plain-English risk signals
  (suspicious keywords, IP-address hosts, link shorteners, risky TLDs, etc.)
- Batch scanning: upload a CSV/TXT file of URLs and get a classification for each
- Scan history log (SQLite) with verdicts and confidence scores
- Model info page showing accuracy, precision, recall, F1-score
- JSON API endpoints (`/api/analyze`, `/api/stats`) for programmatic use
- Live host-based checks (DNS resolution, TLS certificate presence) that
  degrade gracefully to a neutral value when offline

## Architecture

- **Backend:** Python 3 + Flask
- **ML Model:** scikit-learn `RandomForestClassifier` (200 trees, max depth 8)
- **Feature Extraction:** custom lexical + host-based feature engineering (`features.py`)
- **Database:** SQLite (scan history, batch job log)
- **Frontend:** Server-rendered Jinja2 templates + vanilla CSS

## Setup

```bash
cd app
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# One-time: generate the training dataset and train the model
python train_model.py

# Start the web app
python app.py
```

Visit **http://127.0.0.1:5001**.

## About the Training Data

This project ships with a **synthetically generated** dataset (`train_model.py`)
rather than a downloaded one, so the entire pipeline runs fully offline and
reproducibly in any environment. The generator encodes the same statistical
patterns documented in phishing-detection literature (long/obfuscated URLs,
IP-address hosts, suspicious keywords, link shorteners, risky TLDs, "hard"
overlapping examples, and ~2% label noise) to avoid a trivially-separable
toy problem.

**To train on real data instead** (recommended before any production use):
1. Obtain a labelled dataset such as the UCI "Phishing Websites" dataset or a
   PhishTank / OpenPhish export of confirmed phishing URLs, plus a legitimate
   sample (e.g. a top-sites list).
2. Run every URL through `features.extract_all_features()` to build a feature
   matrix using the same `FEATURE_ORDER` schema.
3. Point `train_and_evaluate()` in `train_model.py` at that DataFrame instead
   of `generate_synthetic_dataset()`.

## Project Structure

```
app/
├── app.py                  # Flask application (routes, prediction logic)
├── features.py              # URL feature extraction (24 features)
├── train_model.py           # Dataset generation + model training + evaluation plots
├── requirements.txt
├── model/
│   ├── phishing_model.pkl    # Trained Random Forest model
│   ├── metrics.json          # Accuracy / precision / recall / F1
│   ├── feature_names.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── instance/
│   └── phishing.db           # SQLite database (auto-created)
├── static/css/style.css
└── templates/
    ├── base.html, index.html, result.html
    ├── batch.html, history.html, about.html, error.html
```

## Feature Set (24 features)

| Category | Examples |
|---|---|
| Lexical | URL/hostname/path length, dot/hyphen/digit counts, digit ratio |
| Structural | subdomain count, IP-address host, '@' symbol, port number |
| Keyword-based | presence & count of phishing-associated keywords |
| Domain reputation | suspicious TLDs (.tk, .xyz, .top...), link-shortening services |
| Security | HTTPS usage, "https" token spoofed inside hostname |
| Host-based (live) | DNS resolution success, valid TLS certificate presence |

## Model Performance (on synthetic held-out test set)

- Accuracy: ~97.7%
- Precision: ~97.3%
- Recall: ~98.0%
- F1-score: ~97.7%

See `model/confusion_matrix.png`, `model/roc_curve.png`, and
`model/feature_importance.png` for full evaluation visuals.

## Possible Extensions

- Train on a real-world dataset (UCI Phishing Websites / PhishTank) for production use
- Add a browser extension front-end for real-time URL checking while browsing
- Incorporate visual/screenshot-based detection (logo/favicon similarity to known brands)
- Add WHOIS-based domain-age features
- Ensemble with a gradient-boosted model (XGBoost/LightGBM) for comparison
