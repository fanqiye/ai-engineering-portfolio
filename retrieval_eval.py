"""Offline Chinese BM25 retrieval with citations and dev-calibrated rejection."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time


def tokenize(text):
    result = []
    for word in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.casefold()):
        if "\u4e00" <= word[0] <= "\u9fff":
            result.extend([word] if len(word) == 1 else [word[i:i+2] for i in range(len(word)-1)])
        else:
            result.append(word)
    return result


class Retriever:
    def __init__(self, docs):
        if not docs or len({d["id"] for d in docs}) != len(docs):
            raise ValueError("Knowledge must be nonempty and IDs unique")
        self.docs = docs
        self.counts = [Counter(tokenize(d["title"] + " " + d["text"])) for d in docs]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / len(docs)
        if not self.average:
            raise ValueError("Knowledge has no searchable tokens")
        self.df = Counter()
        for counter in self.counts:
            self.df.update(counter.keys())

    def search(self, query, method="bm25", k=3):
        if method not in ("overlap", "bm25") or k < 1:
            raise ValueError("Invalid method or k")
        words = sorted(set(tokenize(query)))
        scored = []
        # ponytail: linear scan is sufficient for this small knowledge base; index postings at scale.
        for doc, freq, length in zip(self.docs, self.counts, self.lengths):
            score = 0.0
            for word in words:
                tf = freq[word]
                if not tf:
                    continue
                if method == "overlap":
                    score += 1
                else:
                    idf = math.log(1 + (len(self.docs)-self.df[word]+0.5)/(self.df[word]+0.5))
                    score += idf * tf * 2.5 / (tf + 1.5*(0.25+0.75*length/self.average))
            if score > 0:
                scored.append(dict(id=doc["id"], score=score, source=doc["source"], text=doc["text"]))
        return sorted(scored, key=lambda r: (-r["score"], r["id"]))[:k]


def evaluate_rows(retriever, queries, method):
    rows = []
    for query in queries:
        start = time.perf_counter()
        hits = retriever.search(query["query"], method)
        rows.append(dict(**query, retrieved=[h["id"] for h in hits],
                         score=hits[0]["score"] if hits else 0.0,
                         latency_ms=(time.perf_counter()-start)*1000))
    return rows


def accepted(row, threshold):
    return bool(row["retrieved"]) and row["score"] >= threshold


def correct(row, threshold):
    if accepted(row, threshold):
        return row["relevant"] is not None and row["retrieved"][0] == row["relevant"]
    return row["relevant"] is None


def threshold_on_dev(rows):
    candidates = sorted({0.0, *(r["score"] for r in rows),
                         math.nextafter(max(r["score"] for r in rows), math.inf)})
    return max(candidates, key=lambda t: (sum(correct(r, t) for r in rows), t))


def summarize(rows, threshold):
    answerable = [r for r in rows if r["relevant"] is not None]
    unanswerable = [r for r in rows if r["relevant"] is None]
    answered = [r for r in rows if accepted(r, threshold)]
    ranks = [r["retrieved"].index(r["relevant"])+1 if r["relevant"] in r["retrieved"] else 0
             for r in answerable]
    return dict(n=len(rows), answerable=len(answerable), unanswerable=len(unanswerable),
                recall_at_1=sum(rank == 1 for rank in ranks)/len(ranks),
                recall_at_3=sum(rank > 0 for rank in ranks)/len(ranks),
                mrr_at_3=sum(1/rank if rank else 0 for rank in ranks)/len(ranks),
                threshold=threshold, coverage=len(answered)/len(rows),
                accepted_answer_accuracy=sum(correct(r, threshold) for r in answered)/len(answered) if answered else None,
                rejection_specificity=sum(not accepted(r, threshold) for r in unanswerable)/len(unanswerable),
                total_correct_fraction=sum(correct(r, threshold) for r in rows)/len(rows),
                mean_latency_ms=sum(r["latency_ms"] for r in rows)/len(rows))


def load(root):
    docs = json.loads((root / "caigou_knowledge.json").read_text(encoding="utf-8"))
    queries = [json.loads(line) for line in (root / "caigou_queries.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = {d["id"] for d in docs}
    if len({q["id"] for q in queries}) != len(queries) or len({q["query"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs or texts")
    if any(q["relevant"] not in ids | {None} or q["split"] not in ("dev", "test") for q in queries):
        raise ValueError("Invalid query split or relevance ID")
    for split in ("dev", "test"):
        selected = [q for q in queries if q["split"] == split]
        if not any(q["relevant"] is None for q in selected) or not any(q["relevant"] for q in selected):
            raise ValueError("Both answerable and unanswerable examples are required")
    return docs, queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--query")
    args = parser.parse_args()
    docs, queries = load(args.data)
    retriever = Retriever(docs)
    if args.query is not None:
        if not args.query.strip() or len(args.query) > 2000:
            parser.error("Provide 1 to 2,000 characters")
        dev = evaluate_rows(retriever, [q for q in queries if q["split"] == "dev"], "bm25")
        threshold = threshold_on_dev(dev)
        hits = retriever.search(args.query)
        answer = hits[0] if hits and hits[0]["score"] >= threshold else None
        print(json.dumps(dict(status="evidence_found" if answer else "insufficient_evidence",
                              evidence=answer, threshold=threshold), ensure_ascii=False, indent=2))
        return
    report = dict(knowledge_count=len(docs), fixture_origin="AI-authored synthetic queries; not user logs",
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.data.glob("*.json*"))},
        methods={})
    predictions = []
    for method in ("overlap", "bm25"):
        dev = evaluate_rows(retriever, [q for q in queries if q["split"] == "dev"], method)
        test = evaluate_rows(retriever, [q for q in queries if q["split"] == "test"], method)
        threshold = threshold_on_dev(dev)
        report["methods"][method] = dict(dev=summarize(dev, threshold), test=summarize(test, threshold))
        predictions.extend(dict(**r, method=method, accepted=accepted(r, threshold), correct=correct(r, threshold)) for r in test)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "retrieval_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "retrieval_predictions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in predictions)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
