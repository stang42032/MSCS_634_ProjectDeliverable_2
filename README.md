# MSCS 634 — Project Deliverable 2: Regression Modeling and Performance Evaluation

## Dataset

Same dataset as Deliverable 1: the **Diamonds dataset** (53,940 records → 53,772 after cleaning), predicting `price` (USD) from carat, cut, color, clarity, and physical dimensions.

## Feature Engineering

Three engineered features, each directly motivated by a Deliverable 1 EDA finding:

- **`volume = x · y · z`** — consolidates the three highly collinear dimension columns (r > 0.95 with each other and with carat) into a single size measure, reducing multicollinearity for the linear models.
- **`carat_sq = carat²`** — captures the non-linear, accelerating shape of the price–carat relationship observed in the EDA scatter plot.
- **`log_price = log10(price)`** — the modeling target, since raw price is heavily right-skewed and log-price is close to symmetric.
- **Ordinal encoding** of `cut`, `color`, and `clarity` as integers 0…4 / 0…6 / 0…7, preserving their natural quality order (rather than one-hot dummies) so a single coefficient captures "better grade → higher price."

Final feature set: `carat`, `carat_sq`, `volume`, `depth`, `table`, `cut_enc`, `color_enc`, `clarity_enc`. Features were standardized (zero mean, unit variance) before fitting the linear models, which is required for Ridge/Lasso penalties to apply fairly across features of very different scales.

## Models Built

| Model | Type |
|---|---|
| Linear Regression | Unregularized multiple regression (baseline) |
| Ridge Regression (α=1.0) | L2-regularized |
| Lasso Regression (α=0.001) | L1-regularized |
| Random Forest (200 trees, depth 12) | Non-linear benchmark |

All models were trained on an 80/20 train/test split (`random_state=42`) predicting `log_price`, then converted back to dollar scale for interpretable error metrics.

## Evaluation Results (held-out test set)

| Model | R² ($) | RMSE ($) | MAE ($) | 5-fold CV R² (mean ± std, log scale) |
|---|---|---|---|---|
| Linear Regression | 0.9325 | $1,024 | $515 | 0.9735 ± 0.0017 |
| Ridge Regression | 0.9326 | $1,023 | $515 | 0.9734 ± 0.0017 |
| Lasso Regression | 0.9357 | $999 | $515 | 0.9731 ± 0.0014 |
| **Random Forest** | **0.9816** | **$535** | **$281** | **0.9895 ± 0.0003** |

Cross-validation R² is reported on the log-price scale (what the models actually optimize), and is consistently tight across all 5 folds for every model (std ≤ 0.002), confirming the results generalize rather than reflecting one lucky split.

## Which Model Performed Best, and Why

**Random Forest performed best by a clear margin** — roughly half the RMSE of every linear model, and the tightest, most stable cross-validation scores. This confirms the true price relationship has non-linear structure (consistent with the accelerating price/carat curve seen in Deliverable 1) that no amount of linear regularization can fully capture.

Among the three linear models, **Ridge and Lasso barely improved on plain Linear Regression.** This was expected once feature engineering had already collapsed `x`, `y`, `z` into a single `volume` feature — most of the severe multicollinearity that regularization exists to counteract was already resolved before the models were even fit. Lasso's small edge (R² 0.9357 vs. 0.9325) comes from shrinking the weakest features (`depth`, `table`) closer to zero.

## Key Insights

- **Size dominates price**, across every model: `carat`, `carat_sq`, and `volume` are consistently the top-ranked features by both linear coefficient magnitude and Random Forest importance.
- **Grade matters, but conditionally**: `clarity_enc` and `color_enc` carry smaller but consistent positive coefficients once size is controlled for — the same "conditional on carat" effect that explained the Simpson's paradox found in Deliverable 1.
- **`depth` and `table` contribute the least** signal; Lasso shrinks them closest to zero of any feature, matching the EDA finding that these proportions barely correlate with price on their own.
- **Residual analysis** shows the linear models systematically under-predict the most expensive diamonds (mild fanning at high prices even after the log-transform), while the Random Forest's residuals stay tight and evenly scattered — the clearest visual evidence that real non-linearity remains in the price tail.

## Practical Recommendation

For a production pricing tool prioritizing accuracy, the **Random Forest** is the clear choice. For a context requiring interpretability (e.g., explaining a listed price to a customer), **Ridge Regression** is preferable — it is nearly as accurate as plain Linear Regression, more numerically stable, and its standardized coefficients translate directly into statements like "one standard deviation of clarity is worth $X in listed price."

## Challenges and How They Were Addressed

- **Interpreting metrics on a transformed target:** training on `log_price` meant raw R²/RMSE weren't directly meaningful in dollar terms. Addressed by reporting both log-scale metrics (what the model optimizes) and dollar-scale metrics (converting predictions back with `10**log_pred`) side by side.
- **Distinguishing genuine model improvement from noise:** a single train/test split can't tell you if one model's edge over another is real. Addressed with 5-fold cross-validation and boxplots of per-fold R², confirming all four models are stable and the Random Forest's advantage is consistent, not a fluke.
- **Deciding how much regularization would matter:** Ridge/Lasso showed almost no improvement over plain linear regression, which could look like implementation error. Investigating this revealed it as an expected consequence of feature engineering already having removed the multicollinearity regularization is meant to fix — worth noting explicitly rather than treating as a null result.

## Repository Contents

- `MSCS_634_ProjectDeliverable_2.ipynb` — full notebook with commented code, feature engineering, model training, evaluation, and visualizations
- `diamonds.csv` — raw dataset
- `README.md` — this file
