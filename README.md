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

1. **Subsampling introduces seed sensitivity and randomness**
   When `subsample < 1`, the model trains on a different subset of rows each round. Even if the seed is fixed, row shuffling changes which examples are selected. This affects both `tree_method='hist'` and `tree_method='exact'` (because the set of rows differs).

2. **Histogram binning introduces row-order sensitivity (except in `exact`)**
   XGBoost uses different algorithms to decide how to find splits:

   * With `tree_method='hist'` (the default for large data), feature values are discretized into fixed-width histograms as data is read. This process is sensitive to row order and thread scheduling, which influences bin boundaries and the resulting splits.
   * Other methods like `approx` and `gpu_hist` also rely on binning and thus inherit similar row-order dependence.
   * Only `tree_method='exact'` avoids this problem by evaluating all possible split thresholds directly; it is not affected by row order (though still sensitive to subsampling if used).

3. **Histogram binning depends on row order**
   With `tree_method='hist'`, feature values are binned as data is read. Shuffling changes the sequence → changes bin cut-points → changes candidate splits. This happens even with subsampling turned off and fixed seeds.

4. **Split decisions propagate through the tree**
   Small differences early in the tree (due to bins or sampling) amplify through successive splits, leading to substantially different tree structure and predictions.

5. **Parallelism introduces nondeterminism in reductions**
   In multithreaded training, operations like histogram accumulation, feature gain computation, and tie-breaking may execute in different orders depending on thread scheduling. This can result in small floating point discrepancies or tie resolution changes—especially when candidate splits are nearly equivalent.

---

## 3 | Remedies

| Concept                             | Example Implementation                                   | Impact                                                  | Drawbacks                                 |
| ----------------------------------- | -------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------- |
| Fix the seed                        | Set `random_state=...`                                   | Reduces randomness in sampling                          | No effect unless subsampling present      |
| Eliminate subsampling               | Set `subsample=1`, `colsample*=1`                        | Removes stochasticity in data use                       | Slower training; higher overfit risk      |
| Use deterministic tree construction | Use `tree_method='exact'`                                | Fully reproducible split decisions (if subsampling off) | Much slower; infeasible on large datasets |
| Ensembling over multiple fits       | Average predictions across K shuffles                    | Smooths variance; improves stability                    | Higher training + inference cost          |
| Use inherently stable learners      | CatBoost (ordered boosting); LightGBM deterministic mode | Near-zero drift out of the box                          | May require reengineering and tuning      |

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

| Variant                                   | Accuracy | ROC_AUC | Stability_RMSE |
|-------------------------------------------|---------:|--------:|---------------:|
| Single XGB (K=15)                         | 0.9285   | 0.9643  | 0.0313         |
| Ensemble (K=15) × 5                       | 0.9310   | 0.9647  | 0.0072         |
| XGB Random-Forest                         | 0.8957   | 0.9514  | 0.0086         |
| XGB Exact (subsample = 1, colsample = 1)  | 0.9250   | 0.9632  | 0.0000         |


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
