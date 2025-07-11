# StableBoost: Stable XGBoost Predictions Under Data Shuffling

## 1 | Why It Matters

Even with a fixed random seed, **shuffling the order of training rows changes XGBoost’s histogram bins** (`tree_method='hist'`).
Those altered cut-points yield a different forest and, consequently, **different predictions on exactly the same data**.
Symptoms in production:

* Apparent “drift” after a routine retrain
* Flaky regression tests on model outputs
* Spurious monitoring alerts

---

## 2 | Prior Remedies in the Wild

| Concept                             | Example Implementation                                   | Impact                               | Drawbacks                                 |
| ----------------------------------- | -------------------------------------------------------- | ------------------------------------ | ----------------------------------------- |
| Fix the seed                        | Set `random_state=...`                                   | Reduces randomness in sampling       | Doesn't address binning or threading      |
| Eliminate subsampling               | Set `subsample=1`, `colsample*=1`                        | Removes stochasticity in data use    | Slower training; higher overfit risk      |
| Use deterministic tree construction | Use `tree_method='exact'`                                | Fully reproducible split decisions   | Much slower; infeasible on large datasets |
| Ensembling over multiple fits       | Average predictions across K shuffles                    | Smooths variance; improves stability | Higher training + inference cost          |
| Use inherently stable learners      | CatBoost (ordered boosting); LightGBM deterministic mode | Near-zero drift out of the box       | May require reengineering and tuning      |

> ℹ️ The `exact` method performs greedy split finding by checking all possible thresholds for each feature value—no binning or approximation. It is more stable but much slower than the histogram-based default.

---

## 3 | Our Baseline & Metric

**Experimental design**

1. **Fixed train/test split** (75% / 25%)
2. **No resampling** – same rows every time, only shuffled order
3. Fit *K* independent XGB models (different permutations)
4. Evaluate on the held-out test set

**Stability metric**

For test observation *j* and *K* models:

$$
\displaystyle \text{RMSE}_j
    = \sqrt{2\,\text{Var}_{i}\!\bigl(\hat p_{ij}\bigr)}  
\quad\Longrightarrow\quad
\text{MeanRMSE}
  = \frac{1}{N_{\text{test}}}\sum_j \text{RMSE}_j
$$

> Interprets as the **expected RMSE between predictions from two fresh retrains**.

---

## 4 | Illustrative Results (synthetic data, *K = 15*)

| Variant               | Accuracy   | ROC-AUC    | Stability RMSE ↓ |
| --------------------- | ---------- | ---------- | ---------------- |
| Single XGB (baseline) | **0.9290** | **0.9640** | **0.0314**       |
| Ensemble of 15 (avg)  | 0.9280     | 0.9648     | **0.0000**       |
| XGB Random-Forest     | 0.8957     | 0.9518     | **0.0088**       |

*Row-order alone ≈ 3 pp RMSE; bagging drives it to (near) zero.*

---

## 5 | Clean Reproducible Script

Save as **`xgb_stability_demo.py`** and run:

```bash
python xgb_stability_demo.py --n-runs 15 --output results.csv
```

```python
# See full script in prior message
```

---

## 6 | Using This in Practice

1. **Drop-in** your real feature matrix in place of `make_classification`.
2. Tune *K* for runtime vs. stability; RMSE shrinks ∼1/√K.
3. Integrate the metric into CI—fail builds when `Stability_RMSE` exceeds your tolerance (e.g., 0.01).
4. Optionally extend with bootstrap or CV resamples to capture full pipeline variance.
5. For stricter determinism:

   * Set `subsample=1`, `colsample_bytree=1`, and related knobs.
   * Use `random_state` for all randomness.
   * Consider `tree_method='exact'` if dataset is small.

---

A lightweight, reproducible way to quantify—and nearly eliminate—row-order instability in XGBoost pipelines.
