# Quick answers and live verification / 快速回答与真实核查验证

Measured on 2026-10-04 (Asia/Shanghai); documentation updated 2026-10-05.

## Behavior

The Streamlit verification checkbox defaults to off, overriding the graph configuration for that request. Quick answers retain evidence citations and explicitly report that verification did not run. Opt-in verification allows bounded correction. A rewrite replaces the initial answer only after verification passes; otherwise the original answer and its verification record are restored. Request checkpoints preserve already-generated content when a later stage fails or times out.

## Observations

Two questions from the repository's public evaluation index were tested with DeepSeek (`deepseek-chat`) in quick and verification modes. Graph requests took approximately 1.99–5.96 seconds without checking and 5.29–7.58 seconds with checking. First component initialization took approximately 13 seconds separately. These ranges cover two rounds of small fixed-case observations; they are not Streamlit end-to-end measurements or latency percentiles.

During the first run, an initially supported TXT/PDF answer was rewritten into a refusal and then classified as having no claims. The guard was added and tested: a later failed check retained the initial TXT/PDF answer with a failed status. The checker still misjudges this case; retaining content does not establish its correctness or imply a successful check.

The live test temporarily used DeepSeek for both generation and verification. It did not validate the configured SiliconFlow lightweight checker. No private knowledge documents or correction memory were included. Local credentials are ignored by Git and are not included in this report.

## Publication checks

Pre-push regression on 2026-10-05: **39 product regression tests passed**. The historical 2026-09-25 evaluation and its accuracy figures are retained separately; no new 60-question accuracy claim is made.

## Reproduce

```sh
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

Build a knowledge index from public or synthetic TXT/PDF documents as described in the README. Ask the same supported question with verification unchecked, then checked; inspect the evidence, status and stage timings. Test unknown facts separately to ensure that absent evidence produces an explicit limitation/refusal. First-request model loading should be measured separately from warm requests.

The updated regression suite covers rewrite exceptions, rejected rewrites and preservation of generated content after later-stage failure. These tests use fakes and do not consume API credits. The historical 60-question effectiveness evaluation in the README was not rerun and must not be interpreted as the accuracy of this update.

## Remaining limits / 剩余边界

Verifier false positives/negatives and claim-selection drift remain unresolved. Local retrieval is useful for traceable document-grounded answers, but is not a substitute for verified source material, automatic preference learning or production load testing. Web search and the original lightweight checker were not newly validated.
