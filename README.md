# Logistic Regression Analysis
 
A from-scratch implementation of logistic regression using Newton's method (gradient + Hessian), validated against scikit-learn on a real-world Airbnb dataset. Built as part of a Break Through Tech AI Studio "ML Life Cycle: Modeling" lab.
 
## Table of Contents
- [Business Brief](#business-brief)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Analysis](#analysis)
- [AI Usage Reflection](#ai-usage-reflection)
- [How to Run](#how-to-run)
- [Repository Contents](#repository-contents)
## Business Brief
 
**Company & Context:** CoreML is the internal machine learning platform team at a mid-sized tech company. The team builds and maintains custom model implementations that other data science teams across the company rely on in their pipelines.
 
**Business Challenge:** Other teams depend on CoreML's implementations being both *correct* and *fast*. A custom implementation that produces wrong results is a liability — teams building on top of it get bad predictions without knowing why. An implementation that's too slow isn't useful in production. Both correctness and performance must be verified before any new implementation is added to the platform.
 
**Business Goal:** Build a logistic regression implementation the platform can stand behind — one that matches a trusted reference (scikit-learn) and has documented, well-understood run time.
 
**Role & Task:** As a junior ML engineer on CoreML, the assignment was to:
1. Build a logistic regression implementation from scratch and train it on the benchmark dataset.
2. Compare it against scikit-learn's implementation to verify correctness.
3. Benchmark run time against scikit-learn's using the same dataset and document the findings.
## Project Objectives
 
- Implement a `LogisticRegressionScratch` class from first principles using Newton's method (gradient + Hessian-based optimization).
- Train the model to predict whether an Airbnb host is a "superhost" — a binary classification problem.
- Validate correctness by comparing fitted weights/intercept against scikit-learn's `LogisticRegression`.
- Benchmark and document run time against scikit-learn using `%timeit`.
## Dataset
 
- **Source:** `airbnbData_train.csv` — a preprocessed NYC short-term rental ("Airbnb listings") dataset (one-hot encoding, scaling, and imputation already applied).
- **Label:** `host_is_superhost` (binary: `True`/`False`)
- **Features used:**
  - `review_scores_rating`
  - `review_scores_cleanliness`
  - `review_scores_checkin`
  - `review_scores_communication`
  - `review_scores_value`
  - `host_response_rate`
  - `host_acceptance_rate`
## Methodology
 
The `LogisticRegressionScratch` class implements the following components:
 
| Method | Purpose |
|---|---|
| `__init__()` | Sets convergence tolerance and max iterations as stopping criteria |
| `predict_proba(X)` | Computes probabilities via the inverse logit: `P = 1 / (1 + e^(-XW))` |
| `compute_gradient(X, y, P)` | Computes the log-loss gradient: `G = -(y - P) · X` |
| `compute_hessian(X, P)` | Computes the Hessian: `H = (Xᵀ · Q) · X`, where `Q = P(1-P)` |
| `update_weights(X, y)` | Applies the Newton's method update: `w_t = w_(t-1) - H⁻¹ · G` |
| `check_stop()` | Stops when the normalized Euclidean distance between successive weight vectors falls below tolerance |
| `fit(X, y)` | Initializes weights (intercept seeded at the log-odds of the base rate), adds the intercept column, and iterates the update/stop loop until convergence or max iterations |
 
The intercept is absorbed directly into the feature matrix `X` (as an appended column of ones) rather than handled as a separate term, simplifying the gradient and Hessian math.
 
## Results
 
**Correctness:** Weights and intercept produced by `LogisticRegressionScratch` were nearly identical to scikit-learn's `LogisticRegression(C=10**10)` (a high `C` value disables regularization for a fair, apples-to-apples comparison).
 
**Performance:** Benchmarked with `%timeit` on identical training data:
 
| Implementation | Average fit time |
|---|---|
| `LogisticRegressionScratch` (from scratch) | ~55.5 ms |
| scikit-learn `LogisticRegression` | ~117 ms |
 
The from-scratch implementation was roughly **2x faster** than scikit-learn's default solver on this dataset, largely because Newton's method converges in very few iterations for well-behaved, low-dimensional problems like this one.
 
## Analysis
 
During training, the loss function measures how far the model's predicted probabilities are from the true labels. Gradient descent (here, Newton's method) iteratively updates the weights to drive this loss down, using both the gradient (direction of steepest change) and the Hessian (curvature) to take more efficient steps than plain gradient descent. Training stops once successive weight updates change by less than the tolerance threshold, indicating convergence.
 
Comparing the two implementations on both correctness and speed: the nearly identical weights/intercept confirm the from-scratch implementation is mathematically correct, and the faster run time suggests it's a strong candidate for CoreML's platform — it produces trustworthy results without sacrificing (and in this case improving) performance. Before shipping to production, it would still be worth validating on larger and higher-dimensional datasets, since Newton's method's per-iteration cost (inverting the Hessian) scales less favorably as feature count grows.
 
## AI Usage Reflection
 
AI assistance (Google Gemini) was used primarily to debug syntax errors encountered while implementing the gradient and Hessian methods — specifically around correctly passing `X`, `y`, and `P` as arguments rather than mistakenly using `self`, and fixing case-sensitivity issues (`w` vs. `W`). Correctness was verified by comparing the scratch implementation's fitted weights and intercept directly against scikit-learn's output; matching values confirmed the implementation was sound. Going forward, revisiting core concepts (e.g., the purpose of log loss) alongside implementation work would help build clearer intuition while coding, rather than only fixing errors as they arise.

## Next Steps

Perform more testing and explore other regression models
 
## How to Run
 
1. Ensure the following are installed: `pandas`, `numpy`, `scikit-learn`.
2. Place `airbnbData_train.csv` in a `data/` subdirectory relative to the notebook.
3. Open `LogisticRegressionFromScratch.ipynb` in Jupyter and run all cells in order.
4. Review the printed weights/intercept comparison and `%timeit` benchmark output in the final cells.
```bash
pip install pandas numpy scikit-learn
jupyter notebook LogisticRegressionFromScratch.ipynb
```
 
## Repository Contents
 
- `LogisticRegressionFromScratch.ipynb` — main notebook with the full implementation, training, comparison, and analysis
- `data/airbnbData_train.csv` — benchmark dataset (not included; place locally)
- `README.md` — this file
