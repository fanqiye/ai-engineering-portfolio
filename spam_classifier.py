"""Reproducible cross-video spam classification; Python 3.11+ and NumPy only."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import platform
import re
import sys
import time
import urllib.request
import zipfile

import numpy as np

DATA_URL = "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip"
DATA_SHA256 = "bd6182891adb3cfc8334b82c062176dfbebc563bf0ba07e31c2645f916865a0a"
GROUPS = ["Youtube01-Psy.csv", "Youtube02-KatyPerry.csv", "Youtube03-LMFAO.csv",
          "Youtube04-Eminem.csv", "Youtube05-Shakira.csv"]


def normalized(text):
    return " ".join(html.unescape(text).casefold().split())


def tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", normalized(text))


def load_data(path):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError("Dataset checksum mismatch; inspect the source before updating the pin")
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for group in GROUPS:
            for row in csv.DictReader(io.StringIO(archive.read(group).decode("utf-8-sig"))):
                if row["CLASS"] not in ("0", "1") or not normalized(row["CONTENT"]):
                    raise ValueError("Missing text or invalid binary label")
                key = hashlib.sha256(normalized(row["CONTENT"]).encode()).hexdigest()
                rows.append(dict(key=key, text=row["CONTENT"], y=int(row["CLASS"]), group=group))
    labels = defaultdict(set)
    for row in rows:
        labels[row["key"]].add(row["y"])
    conflicts = {key for key, values in labels.items() if len(values) > 1}
    unique = {}
    for row in rows:
        if row["key"] not in conflicts:
            unique.setdefault(row["key"], row)
    clean = list(unique.values())
    splits = [[r for r in clean if r["group"] in groups]
              for groups in (GROUPS[:3], GROUPS[3:4], GROUPS[4:])]
    if any(not split or {r["y"] for r in split} != {0, 1} for split in splits):
        raise ValueError("Every split must contain both classes")
    keys = [set(r["key"] for r in split) for split in splits]
    if any(keys[i] & keys[j] for i in range(3) for j in range(i)):
        raise ValueError("Duplicate leakage across splits")
    return splits, dict(archive_sha256=digest, raw_rows=len(rows), unique_rows=len(clean),
                       conflicting_text_groups=len(conflicts), removed_rows=len(rows)-len(clean),
                       groups=dict(train=GROUPS[:3], validation=GROUPS[3:4], test=GROUPS[4:]),
                       split_counts={name:dict(n=len(s), spam=sum(r["y"] for r in s))
                                     for name, s in zip(("train", "validation", "test"), splits)})


def vocabulary(texts, limit=4000):
    df = Counter()
    for text in texts:
        df.update(set(tokens(text)))
    words = sorted((w for w, n in df.items() if n >= 2), key=lambda w: (-df[w], w))[:limit]
    return {word: i for i, word in enumerate(words)}


def counts(texts, vocab):
    # ponytail: dense arrays suit this 1,956-row corpus; use sparse matrices at larger scale.
    result = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for i, text in enumerate(texts):
        for token, count in Counter(tokens(text)).items():
            if token in vocab:
                result[i, vocab[token]] = count
    return result


def tfidf(x, idf):
    values = np.zeros_like(x)
    np.log(x, out=values, where=x > 0)
    values = (values + (x > 0)) * idf
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def sigmoid(x):
    result = np.empty_like(np.asarray(x, dtype=float))
    positive = x >= 0
    result[positive] = 1 / (1 + np.exp(-x[positive]))
    exp = np.exp(x[~positive])
    result[~positive] = exp / (1 + exp)
    return result


def loss_gradient(x, y, w, b, l2):
    logits = x @ w + b
    error = sigmoid(logits) - y
    loss = np.mean(np.logaddexp(0, logits) - y * logits) + l2 * (w @ w) / 2
    return float(loss), x.T @ error / len(y) + l2 * w, float(error.mean())


def fit_logistic(x, y, steps=500, l2=0.001):
    w, b = np.zeros(x.shape[1]), 0.0
    trace = []
    for step in range(steps):
        loss, dw, db = loss_gradient(x, y, w, b, l2)
        w -= dw
        b -= db
        if step % 50 == 0 or step == steps - 1:
            trace.append(dict(step=step, loss=loss))
    return w, b, trace


def metrics(y, probability, threshold):
    prediction = probability >= threshold
    tp = int(np.sum(prediction & (y == 1)))
    fp = int(np.sum(prediction & (y == 0)))
    fn = int(np.sum(~prediction & (y == 1)))
    tn = int(np.sum(~prediction & (y == 0)))
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(n=len(y), threshold=float(threshold), accuracy=(tp+tn)/len(y),
                precision=precision, recall=recall, f1=2*tp/max(2*tp+fp+fn, 1),
                confusion=dict(tn=tn, fp=fp, fn=fn, tp=tp))


def average_precision(y, score):
    order = np.argsort(-score, kind="stable")
    labels, values = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(values)) + 1, len(y)]
    true_positive = np.cumsum(labels)[ends-1]
    recall = true_positive / max(int(y.sum()), 1)
    return float(np.sum(np.diff(np.r_[0, recall]) * true_positive / ends))


def select_threshold(y, score):
    # Selection uses validation only; prefer the larger threshold when F1 ties.
    return max(np.arange(0.05, 0.951, 0.01), key=lambda t: (metrics(y, score, t)["f1"], t))


def bootstrap_f1(y, probability, threshold):
    rng = np.random.default_rng(20260906)
    values = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), size=len(y))
        values.append(metrics(y[idx], probability[idx], threshold)["f1"])
    return np.quantile(values, [0.025, 0.975]).tolist()


def run(data_path, out):
    start = time.perf_counter()
    splits, audit = load_data(data_path)
    train, valid, test = splits
    vocab = vocabulary([r["text"] for r in train])
    matrices = [counts([r["text"] for r in split], vocab) for split in splits]
    labels = [np.array([r["y"] for r in split]) for split in splits]
    idf = np.log((1 + len(train)) / (1 + np.sum(matrices[0] > 0, axis=0))) + 1
    features = [tfidf(x, idf) for x in matrices]
    w, b, trace = fit_logistic(features[0], labels[0])
    word_counts = np.array([matrices[0][labels[0] == c].sum(axis=0) + 1 for c in (0, 1)])
    log_likelihood = np.log(word_counts / word_counts.sum(axis=1, keepdims=True))
    log_prior = np.log(np.bincount(labels[0], minlength=2) / len(train))
    predictions = {
        "majority": [np.full(len(split), float(labels[0].mean() >= 0.5)) for split in splits],
        "multinomial_nb": [sigmoid((x @ log_likelihood.T + log_prior)[:, 1] -
                                   (x @ log_likelihood.T + log_prior)[:, 0]) for x in matrices],
        "tfidf_logistic": [sigmoid(x @ w + b) for x in features],
    }
    report = dict(protocol="protocol.md", environment=dict(python=sys.version.split()[0],
                  numpy=np.__version__, platform=platform.platform()), data=audit,
                  vocabulary_size=len(vocab), logistic_training=trace, models={})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, probabilities in predictions.items():
        threshold = 0.5 if name == "majority" else select_threshold(labels[1], probabilities[1])
        report["models"][name] = dict(validation=metrics(labels[1], probabilities[1], threshold),
            test=metrics(labels[2], probabilities[2], threshold),
            test_average_precision=average_precision(labels[2], probabilities[2]),
            test_f1_bootstrap_95=bootstrap_f1(labels[2], probabilities[2], threshold))
    report["elapsed_seconds"] = time.perf_counter()-start
    report["limitations"] = ["Historical five-video corpus; not current production traffic",
        "Exact normalized deduplication only; near-duplicate leakage may remain",
        "One held-out video; bootstrap interval does not measure cross-video uncertainty",
        "Validation video selects thresholds, not test; probabilities are not calibrated",
        "AI-assisted educational implementation; not evidence of independent author proficiency"]
    (out / "spam_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    model = dict(vocabulary=vocab, idf=idf.tolist(), weights=w.tolist(), bias=b,
                 threshold=report["models"]["tfidf_logistic"]["test"]["threshold"])
    (out / "spam_model.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    with (out / "spam_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["text_sha256", "label", "logistic_probability", "prediction"])
        for row, prob in zip(test, predictions["tfidf_logistic"][2]):
            writer.writerow([row["key"], row["y"], format(prob, ".12g"), int(prob >= model["threshold"])])
    (out / "spam_split_hashes.json").write_text(json.dumps({name: [r["key"] for r in split]
        for name, split in zip(("train", "validation", "test"), splits)}, indent=2), encoding="utf-8")
    print(json.dumps({"data":audit, "models":report["models"], "seconds":report["elapsed_seconds"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".cache/youtube-spam.zip"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", default="results")
    parser.add_argument("--predict", help="Classify text using results/spam_model.json")
    args = parser.parse_args()
    if args.predict is not None:
        if not args.predict.strip() or len(args.predict) > 20000:
            parser.error("Provide 1 to 20,000 characters")
        model = json.loads((Path(args.out) / "spam_model.json").read_text(encoding="utf-8"))
        x = tfidf(counts([args.predict], model["vocabulary"]), np.array(model["idf"]))
        prob = float(sigmoid(x @ np.array(model["weights"]) + model["bias"])[0])
        print(json.dumps(dict(spam_score=prob, label=int(prob >= model["threshold"]))))
        return
    if args.download and not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=45) as response:
            raw = response.read(2_000_001)
        if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
            raise ValueError("Download checksum mismatch; no data written")
        args.data.write_bytes(raw)
    run(args.data, args.out)


if __name__ == "__main__":
    main()
