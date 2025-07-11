# StableBoost: Stable XGBoost Predictions Under Data Shuffling

## 1 | Why It Matters

Even with a fixed random seed, **shuffling the order of training rows changes XGBoost’s histogram bins** (`tree_method='hist'`).
Those altered cut-points yield a different forest and, consequently, **different predictions on exactly the same data**.
Symptoms in production:

* Apparent “drift” after a routine retrain
* Flaky regression tests on model outputs
* Spurious monitoring alerts

---

# XGBoost Prediction Instability

*A concise technical note*

---

## 1 | Why It Matters

Even with a fixed random seed, **shuffling the order of training rows changes XGBoost’s histogram bins** (`tree_method='hist'`).
Those altered cut-points yield a different forest and, consequently, **different predictions on exactly the same data**.
Symptoms in production:

* Apparent “drift” after a routine retrain
* Flaky regression tests on model outputs
* Spurious monitoring alerts

---

## 2 | Root Causes of Instability

1. **Histogram bin boundaries depend on row order**
   XGBoost’s `tree_method='hist'` builds feature histograms on-the-fly by scanning data in memory. Different row orderings → different bin boundaries → different candidate splits.

2. **Split decisions cascade through the tree**
   Because trees are built top-down, early split differences caused by histogram variation propagate deeply, resulting in structurally different trees.

3. **Subsampling magnifies stochasticity**
   When `subsample < 1`, the row sampling process depends on the order of input data. So even with the same seed, different permutations yield different sample selections → further amplifying tree variation.

4. **Threading and parallel reductions**
   In multithreaded training, small race conditions in histogram construction or split evaluation can interact with row order, adding nondeterminism.

---

## 3 | Remedies

| Concept                             | Example Implementation                                   | Impact                               | Drawbacks                                 |
| ----------------------------------- | -------------------------------------------------------- | ------------------------------------ | ----------------------------------------- |
| Fix the seed                        | Set `random_state=...`                                   | Reduces randomness in sampling       | Doesn't address binning or threading      |
| Eliminate subsampling               | Set `subsample=1`, `colsample*=1`                        | Removes stochasticity in data use    | Slower training; higher overfit risk      |
| Use deterministic tree construction | Use `tree_method='exact'`                                | Fully reproducible split decisions   | Much slower; infeasible on large datasets |
| Ensembling over multiple fits       | Average predictions across K shuffles                    | Smooths variance; improves stability | Higher training + inference cost          |
| Use inherently stable learners      | CatBoost (ordered boosting); LightGBM deterministic mode | Near-zero drift out of the box       | May require reengineering and tuning      |

> ℹ️ The `exact` method performs greedy split finding by checking all possible thresholds for each feature value—no binning or approximation. It eliminates histogram-induced variance, but subsampling can still introduce model differences unless disabled.

---

## 4 | Our Baseline & Metric

**Experimental design**

1. **Fixed train/test split** (75% / 25%)
2. **No resampling** – same rows every time, only shuffled order
3. Fit *K* independent XGB models (different permutations)
4. Evaluate on the held-out test set

**Stability metric**

For test observation *j* and *K* models:

$$\displaystyle \text{RMSE}_j
    = \sqrt{2\,\text{Var}_{i}\!\bigl(\hat p_{ij}\bigr)}  
\quad\Longrightarrow\quad
\text{MeanRMSE}
  = \frac{1}{N_{\text{test}}}\sum_j \text{RMSE}_j$$

> Interprets as the **expected RMSE between predictions from two fresh retrains**.

---

## 5 | Illustrative Results (synthetic data, *K = 15*)

| Variant               | Accuracy   | ROC-AUC    | Stability RMSE ↓ |
| --------------------- | ---------- | ---------- | ---------------- |
| Single XGB (baseline) | **0.9290** | **0.9640** | **0.0314**       |
| Ensemble of 15 (avg)  | 0.9280     | 0.9648     | **0.0000**       |
| XGB Random-Forest     | 0.8957     | 0.9518     | **0.0088**       |

*Row-order alone ≈ 3 pp RMSE; bagging drives it to (near) zero.*

---

## 6 | Clean Reproducible Notebook

* [Notebook](https://github.com/finite-sample/stableboost/blob/main/stableboost.ipynb)

---

## 7 | Using This in Practice

1. **Drop-in** your real feature matrix in place of `make_classification`.
2. Tune *K* for runtime vs. stability; RMSE shrinks ∼1/√K.
3. Integrate the metric into CI—fail builds when `Stability_RMSE` exceeds your tolerance (e.g., 0.01).
4. Optionally extend with bootstrap or CV resamples to capture full pipeline variance.
5. For stricter determinism:

   * Set `subsample=1`, `colsample_bytree=1`, and related knobs.
   * Use `random_state` for all randomness.
   * Consider `tree_method='exact'` if dataset is small.

## 8 | Authors

Victor Shia and Gaurav Sood
