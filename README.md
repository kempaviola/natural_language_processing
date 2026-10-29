# Flagging Toxic Comments for Human Moderators

Capstone project for the Natural Language Processing challenge: a fine-tuned DistilBERT that ranks news comments by how
likely they are to be toxic, so human moderators can read the worst ones first. It is benchmarked against a TF-IDF and
logistic regression baseline and then checked for errors, explanations, identity bias and calibration.

Everything is in one notebook: [nlp_capstone.ipynb](nlp_capstone.ipynb). It contains the code, the figures and the
written report, including the executive summary at the top.

> **Content warning:** the notebook prints real comments, some with slurs, profanity and threats.

## Data

Civil Comments, Borkan, Dixon, Sorensen, Thain & Vasserman (2019), *Nuanced Metrics for Measuring Unintended Bias with
Real Data for Text Classification*, released by Jigsaw under CC0 and loaded from Hugging Face as
[`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments). The notebook downloads it on first
run; the prepared dataset takes about 650 MB in the Hugging Face cache.

## Task

| | |
|---|---|
| Target | toxic (`toxicity >= 0.5`, at least half of the annotators) or non-toxic |
| Input | comment text, English, cleaned lightly (links replaced, whitespace collapsed) |
| Split | the dataset's own split: 50,000 sampled training and 10,000 sampled validation comments, full test split of 97,320 |
| Main metric | PR-AUC for the toxic class, plus F1 at a threshold chosen on validation |
| Hypotheses | H1: DistilBERT beats TF-IDF + LR on PR-AUC. H2: both models flag harmless comments that mention an identity group more often |

## Results

Test set, 97,320 comments (7,777 toxic):

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | Precision at 80% recall |
|---|---|---|---|---|---|---|
| Always "non-toxic" | 0.920 | 0.000 | 0.000 | 0.000 | 0.080 | 0.080 |
| TF-IDF + logistic regression | 0.923 | 0.518 | 0.591 | 0.552 | 0.588 | 0.292 |
| **DistilBERT, fine-tuned** | **0.945** | **0.650** | **0.668** | **0.659** | **0.728** | **0.498** |

H1 is supported: DistilBERT beats the baseline on every measure, with a bootstrap 95% interval for the F1 gap of
0.098 to 0.115. H2 is also supported, and the fairness criterion fails. Harmless comments that mention black, white,
Muslim or LGBTQ people are flagged 3 to 4 times as often as harmless comments in general, and DistilBERT scores
harmless sentences like "I am a black person." above its threshold.

Recommendation: use the model only to sort a moderation queue in a supervised pilot, never to remove comments
automatically, and reduce the identity bias before any wider use.

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then open `nlp_capstone.ipynb` in Jupyter or VS Code, choose the `.venv` Python as the kernel and run all cells.

The first run fine-tunes DistilBERT, which takes 40 to 50 minutes on an Apple M1 Pro, and saves it to
`models/distilbert_toxicity/`. Later runs load the saved model. Delete that folder to train again. The model folder is
not part of the submission (about 270 MB).

## Files

| File | Content |
|---|---|
| `nlp_capstone.ipynb` | code, figures and written report |
| `requirements.txt` | pinned package versions, including the spaCy English model |
| `results.json` | configuration, versions and every number quoted in the report |
| `training_log.json` | DistilBERT training and validation log, used for the training curves |
| `test_predictions.parquet` | test comments with labels and both models' scores |
| `figures/` | every figure in the notebook as PNG |
