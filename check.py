"""One independent runnable check: gradient, metrics, split isolation and retrieval."""
import hashlib
import json
from pathlib import Path

import numpy as np
from spam_classifier import loss_gradient, metrics, average_precision, tfidf, counts, vocabulary
from retrieval_eval import Retriever, threshold_on_dev, correct, load


def main():
    x = np.array([[1., 2.], [0., 1.], [3., 0.]])
    y = np.array([1, 0, 1])
    w, b, eps = np.array([.2, -.4]), .1, 1e-6
    _, grad, bias_grad = loss_gradient(x, y, w, b, .03)
    for i in range(2):
        delta = np.eye(2)[i] * eps
        numeric = (loss_gradient(x, y, w+delta, b, .03)[0]-loss_gradient(x, y, w-delta, b, .03)[0])/(2*eps)
        assert abs(numeric-grad[i]) < 1e-7
    numeric_b = (loss_gradient(x, y, w, b+eps, .03)[0]-loss_gradient(x, y, w, b-eps, .03)[0])/(2*eps)
    assert abs(numeric_b-bias_grad) < 1e-7
    labels = np.array([1, 0, 1, 0])
    score = np.array([.9, .8, .7, .1])
    assert abs(average_precision(labels, score)-5/6) < 1e-12
    assert average_precision(labels, np.ones(4)) == .5
    assert metrics(labels, score, .75)["confusion"] == dict(tp=1, fp=1, fn=1, tn=1)
    vocab = vocabulary(["cat cat", "cat dog", "dog"])
    assert "validationsecret" not in vocab
    assert not counts(["validationsecret"], vocab).any()
    assert np.isfinite(tfidf(counts([""], vocab), np.ones(len(vocab)))).all()
    docs = [dict(id="a", title="", text="cat cat", source="local:a"),
            dict(id="b", title="", text="dog dog", source="local:b")]
    retriever = Retriever(docs)
    hit = retriever.search("cat")[0]
    import math
    assert hit["id"] == "a" and abs(hit["score"]-math.log(2)*5/3.5) < 1e-12
    assert not retriever.search("unknown") and not retriever.search("")
    dev = [dict(relevant="a", retrieved=["a"], score=5.),
           dict(relevant=None, retrieved=["a"], score=1.)]
    threshold = threshold_on_dev(dev)
    assert all(correct(row, threshold) for row in dev)
    assert not correct(dict(relevant=None, retrieved=["a"], score=6.), threshold)
    load(Path(__file__).parent / "data")
    root = Path(__file__).parent
    manifest = json.loads((root / "results/run_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        content = (root / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == expected, f"Stale manifest hash: {name}"
    print("CHECK_OK: gradients, metrics, retrieval, fixture schema and manifest hashes")


if __name__ == "__main__":
    main()
