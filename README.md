# AI Engineering Portfolio

Two reproducible, CPU-only learning projects connecting content operations with
text classification and document retrieval. Maintained by
[fanqiye](https://github.com/fanqiye); related product:
[Caigou Desktop Pet](https://github.com/fanqiye/caigou-desktop-pet).

**Status:** reproducible educational baselines, not production systems.

## Run

Python 3.11+; only NumPy is required. No API key, paid inference or GPU.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python check.py
.venv\Scripts\python spam_classifier.py --download
.venv\Scripts\python retrieval_eval.py
.venv\Scripts\python spam_classifier.py --predict "Subscribe to my channel and win a free prize"
.venv\Scripts\python retrieval_eval.py --query "睡眠能恢复多少体力？"
```

On macOS/Linux use `.venv/bin/python` for the same commands. Run from the
repository root. `--download` fetches a checksum-pinned UCI archive into
`.cache/`; subsequent runs work offline. Raw comments are not included here.
Experiments write to `results/`; use `--out .cache/my-results` to preserve the
committed reference outputs. Inference reads the model from the chosen output
directory; a reference model is included.

## 1  Cross-video YouTube spam classification

The question is whether a text-only classifier transfers to an unseen video.
The dataset has 1,956 comments from five historical videos. HTML/case/whitespace
normalization and exact deduplication leave 1,740 comments: 1,029 train, 400
validation and 311 test. Training uses Psy, KatyPerry and LMFAO; validation uses
Eminem; Shakira is the fixed test video. Author, timestamp and video identifiers
are excluded from model features.

The implementation fits 1,033 unigram features and IDF using training data only.
It compares a majority baseline, multinomial Naive Bayes and TF-IDF logistic
regression. Logistic regression uses NumPy gradients with a finite-difference
check. Each learned classifier selects its decision threshold using validation
F1, then the fixed threshold is evaluated on test.

| Method | Test precision | Test recall | Test F1 | Test accuracy | Average precision |
|---|---:|---:|---:|---:|---:|
| Majority | 0.000 | 0.000 | 0.000 | 0.553 | 0.447 |
| Multinomial NB | 1.000 | 0.791 | 0.884 | 0.907 | 0.934 |
| TF-IDF logistic | 0.991 | 0.791 | 0.880 | 0.904 | 0.963 |

The more complex model did **not** beat Naive Bayes on test F1. Logistic ranking
has higher average precision, but its validation-selected decision threshold
still misses 29 of 139 spam comments and incorrectly flags one legitimate
comment. Keep both facts when discussing the model. Logistic F1's comment-level
bootstrap 95% interval is approximately [0.835, 0.921]; it does not describe
uncertainty across videos.

Evidence: [metrics](results/spam_metrics.json),
[all test predictions](results/spam_predictions.csv),
[split hashes](results/spam_split_hashes.json),
[saved model](results/spam_model.json).

## 2  Desktop-pet knowledge retrieval and rejection

An offline retriever searches 20 paraphrased knowledge entries from the existing
pet repository and returns an evidence excerpt with its source URL. It compares
binary token overlap with BM25 using Chinese character bigrams and Latin words.
This is a separate CLI experiment; it has not been integrated into the WPF pet.
It does not call or evaluate an LLM, generate answers, or measure hallucinations.

The 76 queries are synthetic fixtures: 26 dev and 50 test. The test
set contains 40 answerable and 10 unanswerable questions. Relevance is relative
to this 20-entry knowledge base, not every fact in the upstream code. Thresholds
are selected on dev only. Dev and test share topics, so these are modest
in-domain checks, not evidence of general retrieval ability.

| Method | Recall@1 | Recall@3 | MRR@3 | Answer coverage | Accepted accuracy | Rejection specificity |
|---|---:|---:|---:|---:|---:|---:|
| Token overlap | 0.800 | 0.975 | 0.879 | 0.740 | 0.811 | 0.800 |
| BM25 | 0.850 | 0.950 | 0.900 | 0.820 | 0.805 | 0.600 |

BM25 improves top-1 ranking but has worse top-3 recall and rejects only 6 of 10
unanswerable questions. It returns an irrelevant source for questions such as
“使用哪个模型生成回复？”; a high lexical score is not evidence that a question is
answerable. Another failure, “夜里最长会睡多长时间？”, misses the sleep entry because
the bigram tokenizer cannot connect the paraphrase to the right wording.

The next experiment should collect a new, separately labeled set of questions
and compare a semantic retriever or a dedicated answerability model. Do not
retune these test fixtures and report the result as untouched test performance.

Evidence: [metrics](results/retrieval_metrics.json),
[all test cases and failures](results/retrieval_predictions.jsonl),
[knowledge](data/caigou_knowledge.json), [queries](data/caigou_queries.jsonl).

## Protocol and checks

[protocol.md](protocol.md) records the choices made before the first benchmark
run. [check.py](check.py) is the single runnable correctness check: numerical
gradient, tied-score average precision, confusion matrix, out-of-vocabulary
handling, BM25 formula, rejection behavior and fixture validation. The data
loader also enforces disjoint normalized text hashes across splits. GitHub
Actions runs the small check and offline retrieval pipeline on each change;
it does not train the classifier or download the external corpus.

The original pet passed its existing `tests/test-caigou-dynamic.ps1` check at
commit `048742b363227b2ae768822f0f79bf06cf673163` during this work. Its actual
behavior is WPF interaction, state persistence and randomized rule scheduling;
there is no trained neural policy in that project.

## Data and attribution

- Alberto, T. & Lochter, J. (2015). *YouTube Spam Collection*. UCI Machine
  Learning Repository. https://doi.org/10.24432/C58885 . Dataset license:
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Dataset archive SHA-256:
  `bd6182891adb3cfc8334b82c062176dfbebc563bf0ba07e31c2645f916865a0a`.
- Pet facts are paraphrased from the owner's desktop-pet README; each knowledge
  entry records a pinned source URL.
  No upstream artwork, state files or application source is redistributed here.
- Only normalized comment hashes, labels and numeric predictions are published.
  Exact deduplication does not remove semantic or near duplicates. Historical
  music-video comments are not representative of current content moderation.

See [LEARNING.md](LEARNING.md) for a Chinese walkthrough and suggested extensions.

## Development note

The initial implementation and synthetic retrieval fixtures were developed with
Codex assistance. The reported claims are limited to behavior that can be
reproduced from the committed code and data.
