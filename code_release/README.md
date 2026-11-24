# Harm prioritisation of darknet market listings

Code for the paper *Where harm concentrates online: Harm prioritisation of
darknet market listings as digital guardianship*.

The repository contains the rule-based harm scoring, the descriptive
analysis, the multimodal harm-scoring model and its evaluation. It does not
contain the crawler, market addresses or access configuration.

## Data

The labelled dataset (3,464 listings from ten darknet markets, collected
22–25 February 2026) and the product images are not distributed with this
repository. They are available to qualified researchers on request under a
data use agreement. Vendor pseudonyms are removed from the shared data.

Each record in `Final_Labelled_Dataset.json` has the fields `product_id`,
`market_name`, `category`, `class`, `keyword`, `title`, `price`,
`harm_score` and `local_image_path`.

## Requirements

Python 3.12. Install the packages with

    pip install -r requirements.txt

A CUDA GPU is recommended for `cnn_embeddings.py` (the paper used an
NVIDIA RTX 3060); the other scripts run on a CPU in a few minutes.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Stage 1 rule-based harm scoring | `src/harm_scorer.py products.json OUT_DIR` | harmful / non-harmful / review files |
| 2. Stage 1 versus final scores | `src/stage_comparison.py DATA_JSON` | changed scores, product-level scoring summary |
| 3. Descriptive harm profile and concentration (RQ1) | `src/descriptive_stats.py DATA_JSON OUT_DIR` | `product_classes.csv`, `markets.csv`, concentration shares |
| 4. Train/validation/test splits | `src/make_splits.py DATA_JSON splits.json` | `splits.json` (also included) |
| 5. CNN fine-tuning and image embeddings | `src/cnn_embeddings.py DATA_JSON splits.json EMB_DIR` | `embeddings_seed{0,1,2}.npy` |
| 6. Ridge model, ablation, robustness check | `src/ridge_evaluation.py DATA_JSON splits.json EMB_DIR results.json` | `results.json` |
| 7. Concentration and leave-one-market-out | `src/concentration_lomo.py DATA_JSON` | `concentration_lomo_results.json` |
| 8. Figures | `src/plot_figures.py results.json FIG_DIR` | predicted-vs-assigned plot, ablation chart |

Steps 5 to 7 reproduce the model results (RQ2) reported in the paper. The outputs
reported in the paper are in `results/`.

## Notes on reproducibility

- **Stage 1 routing counts.** The counts of listings routed as harmful,
  non-harmful and for review that are reported in the paper were produced by
  running `harm_scorer.py` on the raw parsed listings, before market and
  category names were harmonised. Running it on the final dataset gives
  slightly different routing counts. The raw parsed listings contain market
  URLs and vendor information and are not distributed; a version without these
  fields can be requested together with the labelled dataset.
- **Stage 2.** Final scores were assigned manually at the level of product
  keywords; `stage_comparison.py` reports how they differ from Stage 1.
- **CNN fine-tuning.** `cnn_embeddings.py` fixes all random seeds and uses
  deterministic cuDNN settings, but GPU computations are not bit-for-bit
  reproducible across hardware and library versions. Re-training the network
  can change the reported metrics in the third decimal place. The ridge models
  are deterministic given the embeddings.

## Evaluation protocol

One random 70/15/15 split is shared by every component, so no test listing
is used to fit any transformation or model. The TF-IDF vocabulary, encoders,
price scaler, CNN fine-tuning, standardisation and PCA are fitted on the
training data. The CNN uses the validation set for early stopping. The final
ridge model is refitted on training plus validation data and evaluated once on
the test set. The procedure is repeated for split seeds 0, 1 and 2.

## Licence

MIT (see `LICENSE`).
