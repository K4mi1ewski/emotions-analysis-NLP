# emotions-analysis-NLP

Comparison of **BERT-style fine-tuning** and **GPT-style zero-shot prompting** for fine-grained
emotion classification on the **GoEmotions** dataset. Project for the Artificial Neural Networks
(SSN) course at AGH.

## Task

GoEmotions (`simplified`) is multi-label, so it is reduced to **single-label classification over
28 classes** (27 emotions + `neutral`) by keeping only examples with exactly one emotion:
36 308 train / 4 548 validation / 4 590 test.

## Models

| Type | Models | Approach |
|------|--------|----------|
| BERT | `distilbert-base-uncased`, `roberta-base`, `albert-base-v2` | fine-tuned 28-class classifiers (3 epochs, lr 2e-5, best checkpoint by macro F1) |
| GPT  | `unsloth/llama-3-8b-Instruct-bnb-4bit`, `unsloth/gemma-2-9b-it-bnb-4bit` | zero-shot — the prompt lists the 28 allowed labels, the free-text answer is mapped back to a class id |

Both families feed the same metric functions, so the numbers are directly comparable.
BERT models are evaluated on the full test set, GPT models on the first `GPT_EVAL_N` examples
(1 000 by default — zero-shot inference is slow).

## Results

| Model | Accuracy | Macro F1 | Weighted F1 | Cohen's κ | Latency (s/example) |
|---|---|---|---|---|---|
| RoBERTa | 0.620 | **0.523** | 0.613 | 0.556 | 0.0020 |
| ALBERT | **0.636** | 0.491 | 0.619 | 0.559 | 0.0031 |
| DistilBERT | 0.630 | 0.484 | 0.618 | **0.561** | **0.0010** |
| Gemma 2 (zero-shot) | 0.239 | 0.234 | 0.242 | 0.192 | 1.96 |
| Llama 3 (zero-shot) | 0.227 | 0.197 | 0.203 | 0.179 | 1.28 |

Fine-tuned encoders beat zero-shot LLMs by a wide margin and are ~1000× faster at inference;
McNemar's test confirms the BERT↔GPT gaps are significant (p ≪ 0.05) while the differences
*within* each family are not. Collapsing the 28 emotions into 4 sentiment groups
(positive / negative / ambiguous / neutral) lifts accuracy to ~0.72 for BERT and ~0.48 for GPT,
showing many errors are confusions between neighbouring emotions
(`annoyance` ↔ `anger`, `approval` → `neutral`). Top-3 accuracy for the BERT models is ~0.86.

## Evaluation

Beyond the summary table, the notebook reports: 28×28 confusion matrices, per-emotion F1,
full `classification_report`, Cohen's κ, hierarchical (sentiment-group) evaluation, top-k accuracy,
most-confused emotion pairs, pairwise inter-model agreement + McNemar's test,
quality-vs-latency trade-off, and F1 vs class frequency.

## Files

- [ssn_emotion_analysis.ipynb](ssn_emotion_analysis.ipynb) — full notebook with saved outputs and plots
- [ssn_emotion_analysis_clean.ipynb](ssn_emotion_analysis_clean.ipynb) — same notebook, outputs cleared

## Running

Built for **Google Colab** with a GPU runtime (T4). `Runtime → Change runtime type → GPU`, then run
all cells. Control flags in the config cell:

- `MODELS_DIR` — where models and results are cached (Google Drive on Colab)
- `FORCE_RETRAIN` — `False` reuses cached models/results, `True` retrains from scratch
- `GPT_EVAL_N` — number of test examples scored by the GPT models (`None` = full test set)

Dependencies (installed by the first cell): `transformers`, `datasets`, `evaluate`, `accelerate`,
`bitsandbytes`, `sentencepiece`, `scikit-learn`, `seaborn`, `statsmodels`.
