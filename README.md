# StableBoost: Stable XGBoost Predictions Under Data Shuffling

## 1 | Why It Matters

Even with a fixed random seed, **shuffling the order of training rows changes XGBoost’s histogram bins** (`tree_method='hist'`).
Those altered cut-points yield a different forest and, consequently, **different predictions on exactly the same data**.
Symptoms in production:

* Apparent “drift” after a routine retrain
* Flaky regression tests on model outputs
* Spurious monitoring alerts

---

## 2 | Root Causes of Instability

1. **Multi-thread histogram binning is row-order sensitive**  
   When `tree_method='hist'` **and** `n_jobs > 1`, each thread builds a local
   quantile sketch on its chunk of rows; merging those sketches makes the final
   bin boundaries depend on how the chunks were formed—hence on row order.
   Single-thread `hist` and `tree_method='exact'` avoid this effect.
   
2. **Row subsampling amplifies sensitivity**  
   With `subsample < 1`, every boosting round trains on only a sample of rows.  
   Shuffling the dataset changes which rows fall into that sample—even under a
   fixed `random_state`—so the gradient seen by each new tree differs.

3. **Column subsampling is a smaller, second-order factor**  
   With `colsample_bytree < 1`, each tree sees a random subset of features.
   Different feature subsets nudge split choices; the resulting drift is
   typically an order of magnitude smaller than the first two causes, but still
   measurable.

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
