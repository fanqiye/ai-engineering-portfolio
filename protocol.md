# Experiment protocol

Defined on 2026-09-06 before the first full dataset run. This is a public,
AI-assisted learning portfolio, not employment or an independently completed
research contribution. No paid model APIs are needed.

## YouTube spam classification

- Source: UCI YouTube Spam Collection, Alberto and Lochter (2015),
  https://doi.org/10.24432/C58885 ; CC BY 4.0.
- Use only CONTENT as a feature and CLASS as the label. Never use author,
  comment identifier, date, or video identity as model features.
- Normalize HTML entities, case and whitespace; remove all conflicting-label
  duplicate groups, then keep the first occurrence of each normalized text in
  source-file order. Report removal counts. This cannot remove near duplicates.
- Split by video: Psy, KatyPerry and LMFAO for training; Eminem for validation;
  Shakira for final testing. These choices are fixed before measuring outcomes.
- Compare training-majority, Laplace-smoothed multinomial Naive Bayes and
  TF-IDF logistic regression. Fit vocabulary and IDF on training only.
- Maximum 4,000 unigram features, minimum document frequency 2. Logistic
  regression: zero initialization, 500 full-batch steps, learning rate 1,
  L2 strength 0.001, no bias regularization, no class reweighting.
- Choose each learned model's threshold on validation F1 only, grid 0.05 to
  0.95 in increments of 0.01. Break ties toward the larger threshold.
- Report all models, including underperforming ones. Report held-out precision,
  recall, F1, accuracy, confusion matrix and stepwise average precision.
- F1 interval: 1,000 bootstrap resamples of test comments, fixed seed 20260906.
  It describes this video only; comments need not be independent and this does
  not establish performance across videos or platforms.
- Record dataset checksum, versions, split text hashes and all test predictions.
  Do not publish raw comments or author names in the repository.

## Desktop pet document retrieval

- Knowledge entries are paraphrased from the owner's public desktop-pet README
  and code at commit 048742b363227b2ae768822f0f79bf06cf673163.
- The new retriever is a separate experiment, not integrated into the WPF app.
- Compare binary token overlap and BM25 (k1=1.5, b=0.75). Chinese character
  bigrams plus Latin word tokens; no synonym tuning after test inspection.
- Queries and relevance labels are AI-authored synthetic fixtures, not user
  traffic. Freeze the fixtures before first run; dev and test have different
  queries but share knowledge topics.
- Report Recall@1, Recall@3 and MRR@3 on answerable queries. Each query has one
  primary relevant document, so Recall@k equals Hit@k in this fixture.
- Calibrate a rejection threshold on dev examples only; choose the threshold
  that maximizes the fraction of correct top-1 answers plus correct rejections.
  Report answer coverage, accepted-answer accuracy and rejection specificity
  separately on test. No LLM generation and no claim to evaluate hallucination.
- A single small runnable check validates algorithm invariants independently
  of benchmark performance. There is no benchmark-score pass threshold.
