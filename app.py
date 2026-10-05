"""
Phishing Detection System
--------------------------
A Flask web application that analyses a URL and classifies it as
"Phishing" or "Legitimate" using a trained Random Forest model over a
24-dimension lexical/host-based feature vector (see features.py).

Run:
    pip install -r requirements.txt
    python train_model.py     # one-time: generates dataset + trains model
    python app.py

Then visit http://127.0.0.1:5000
"""

import os
import io
import csv
import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    jsonify, send_file, g
)

from features import extract_all_features, explain_features, FEATURE_ORDER

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
DATABASE = BASE_DIR / "instance" / "phishing.db"
os.makedirs(DATABASE.parent, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("PHISHING_APP_SECRET", "dev-secret-change-me")

# --------------------------------------------------------------------------
# Load model + metadata (fail gracefully with a clear message if missing)
# --------------------------------------------------------------------------
MODEL = None
METRICS = {}
try:
    with open(MODEL_DIR / "phishing_model.pkl", "rb") as f:
        MODEL = pickle.load(f)
    with open(MODEL_DIR / "metrics.json") as f:
        METRICS = json.load(f)
except FileNotFoundError:
    MODEL = None


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DATABASE)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            url          TEXT NOT NULL,
            hostname     TEXT,
            verdict      TEXT NOT NULL,
            confidence   REAL NOT NULL,
            reasons      TEXT,
            scanned_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS batch_jobs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            filename     TEXT,
            total_urls   INTEGER,
            phishing_count INTEGER,
            legit_count  INTEGER,
            created_at   TEXT NOT NULL
        );
        """
    )
    db.commit()
    db.close()


# --------------------------------------------------------------------------
# Core prediction logic
# --------------------------------------------------------------------------

def predict_url(url, live_checks=True):
    if MODEL is None:
        raise RuntimeError("Model not found. Run `python train_model.py` first.")

    feats, hostname, scheme = extract_all_features(url, live_checks=live_checks)
    vector = [[feats[f] for f in FEATURE_ORDER]]
    proba = MODEL.predict_proba(vector)[0]
    pred = MODEL.predict(vector)[0]

    verdict = "Phishing" if pred == 1 else "Legitimate"
    confidence = round(float(max(proba)) * 100, 2)
    reasons = explain_features(feats)

    return {
        "url": url,
        "hostname": hostname,
        "verdict": verdict,
        "is_phishing": bool(pred == 1),
        "confidence": confidence,
        "phishing_probability": round(float(proba[1]) * 100, 2),
        "legit_probability": round(float(proba[0]) * 100, 2),
        "features": feats,
        "reasons": reasons,
    }


def log_scan(result):
    db = get_db()
    db.execute(
        "INSERT INTO scans (url, hostname, verdict, confidence, reasons, scanned_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (result["url"], result["hostname"], result["verdict"], result["confidence"],
         json.dumps(result["reasons"]), datetime.utcnow().isoformat()),
    )
    db.commit()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def index():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM scans").fetchone()["c"]
    phishing = db.execute("SELECT COUNT(*) c FROM scans WHERE verdict='Phishing'").fetchone()["c"]
    return render_template("index.html", model_loaded=MODEL is not None,
                            metrics=METRICS, total=total, phishing=phishing)


@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.form.get("url", "").strip()
    if not url:
        flash("Please enter a URL to analyze.", "danger")
        return redirect(url_for("index"))
    if MODEL is None:
        flash("Model not trained yet. Run train_model.py first.", "danger")
        return redirect(url_for("index"))

    try:
        result = predict_url(url, live_checks=True)
    except Exception as e:
        flash(f"Could not analyze URL: {e}", "danger")
        return redirect(url_for("index"))

    log_scan(result)
    return render_template("result.html", result=result)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400
    try:
        result = predict_url(url, live_checks=data.get("live_checks", True))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    log_scan(result)
    return jsonify(result)


@app.route("/history")
def history():
    db = get_db()
    rows = db.execute("SELECT * FROM scans ORDER BY scanned_at DESC LIMIT 200").fetchall()
    return render_template("history.html", rows=rows)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", results=None)

    file = request.files.get("csv_file")
    if not file or file.filename == "":
        flash("Please upload a CSV file with one URL per line (optionally with a header).", "danger")
        return redirect(url_for("batch"))
    if MODEL is None:
        flash("Model not trained yet. Run train_model.py first.", "danger")
        return redirect(url_for("batch"))

    content = file.read().decode("utf-8", errors="ignore")
    urls = []
    for line in content.splitlines():
        line = line.strip().strip(",")
        if not line or line.lower() in ("url", "urls"):
            continue
        urls.append(line.split(",")[0])

    results = []
    phishing_count = 0
    for u in urls[:500]:  # safety cap
        try:
            r = predict_url(u, live_checks=False)  # batch: skip live checks for speed
            log_scan(r)
            if r["is_phishing"]:
                phishing_count += 1
            results.append(r)
        except Exception:
            continue

    db = get_db()
    db.execute(
        "INSERT INTO batch_jobs (filename, total_urls, phishing_count, legit_count, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (file.filename, len(results), phishing_count, len(results) - phishing_count,
         datetime.utcnow().isoformat()),
    )
    db.commit()

    return render_template("batch.html", results=results,
                            phishing_count=phishing_count, legit_count=len(results) - phishing_count)


@app.route("/about")
def about():
    return render_template("about.html", metrics=METRICS)


@app.route("/api/stats")
def api_stats():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM scans").fetchone()["c"]
    phishing = db.execute("SELECT COUNT(*) c FROM scans WHERE verdict='Phishing'").fetchone()["c"]
    return jsonify({
        "total_scans": total,
        "phishing_detected": phishing,
        "legitimate": total - phishing,
        "model_metrics": METRICS,
    })


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page Not Found"), 404


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5001)
