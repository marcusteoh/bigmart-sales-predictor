# Defence Brief — Q&A and Case Challenge Preparation

*Working notes for the demo. Not a submission deliverable — this exists so that every claim in
the notebook and pitch can be defended, and so that you are never answering a technical
question for the first time while standing up.*

The rubric awards 8% for **"insightful, critical comparison of solution and alternatives"** and
**"clear and technically accurate justification"**. Those marks are won by knowing *why* each
choice was made and what the alternative would have cost — not by knowing more numbers.

---

## 1. The five questions most likely to be asked

### "Your R² is only 0.61. Isn't that a weak model?"

No — and the framing is the trap. Judge it against the alternatives, not against 1.0:

- Predicting the average for everything gives R² ≈ 0 and a typical miss of **$1,322**.
- This model's typical miss is **~$720** — it removes about **45%** of the forecast error.
- The ceiling is set by the data, not the algorithm. The dataset has no footfall, no promotion
  calendar, no competitor pricing, no stock-on-hand. Two products identical on all 9 available
  fields genuinely sold different amounts, and no model can resolve that.

Evidence that we are near the data's ceiling: four structurally different algorithms
(linear, single tree, bagging, boosting) all converge to R² ≈ 0.56–0.62. When very different
model families agree, the limit is the information in the features.

**The honest close:** "0.61 is what this data supports. Getting to 0.8 needs new data, not a
better algorithm — and I can tell you exactly which data."

### "Why Gradient Boosting when Random Forest scored better on your test set?"

This is the strongest question available, and Section 9.4 answers it deliberately.

- **Selecting on the test set destroys the test set.** If you pick the model that scores best on
  the test split, that score is the maximum of two noisy numbers and is biased upward. It stops
  being an unbiased estimate of future performance — which is the only reason to hold it out.
- Selection was made on **cross-validation**, where GB leads, and where the two were compared on
  identical folds with a confidence interval on the difference.
- The RF test advantage is a few dollars of RMSE on 1,705 rows — inside the noise of which rows
  happened to land in the split.
- Tie-breakers, both measured not asserted: GB's artifact is far smaller and faster to load, and
  the prediction intervals are GB quantile models with the same hyperparameters, so point
  estimate and interval come from one coherent family.

**Say this:** "I chose before looking at the test set, which is the whole point of holding one out."

### "How do I know the model isn't just memorising?"

Three independent pieces of evidence:

1. **Train/test gap** (Section 7.2). The Decision Tree gets 0.0 training RMSE and 1,530 on test —
   that is what memorisation looks like. Tuned GB's gap is small.
2. **Repeated cross-validation** (Section 7.3) — 15 separate train/test splits, not one lucky one,
   with confidence intervals.
3. **No identifiers as features.** `Item_Identifier` and `Outlet_Identifier` are dropped
   deliberately; keeping them would let the model memorise individual products and stores.

### "What happens when BigMart opens a new store?"

Answer honestly — this is a known limitation (Section 12.2), and admitting it scores better than
bluffing:

- All 10 outlets appear in both train and test, because the split is row-wise. So the model is
  validated for **known** stores and products, which matches the stated use case: planning stock
  across the existing estate.
- For a genuinely new outlet, the model would rely on `Outlet_Type`, `Outlet_Size`,
  `Outlet_Location_Type` and age — which is precisely the useful part, since those are the
  features that carry the signal. But it has **not been validated** for that.
- The correct test is a grouped split holding out whole outlets. With only 10 outlets that
  estimate would be very high-variance, which is why it is named as future work rather than
  quietly attempted.

### "Should we raise prices, since MRP drives sales?"

**No — and this is the most important trap in the whole project.** The model learns
*correlation*. Expensive products sell more revenue per unit in this data, but price is not
randomly assigned: premium products differ from cheap ones in brand, category and placement in
ways the model cannot see.

The what-if chart in the app is explicitly labelled a **model sensitivity curve, not a pricing
recommendation**. Establishing a causal price effect needs an experiment — a randomised price
test across comparable stores.

**This answer alone demonstrates the "assessing alternative solutions" criterion.**

---

## 2. Case-challenge scenarios

### "Cut your feature count in half — the data collection is too expensive."

Already answered by the evidence: permutation importance shows `Item_MRP` and `Outlet_Type`
carry nearly all the signal, while `Item_Weight`, `Item_Type` and `Item_Fat_Content` score at
essentially zero. A two-feature model would lose very little.

Note the nuance: we *kept* them anyway, because they are free to collect at prediction time — a
category manager already knows the product type. There was no accuracy-versus-cost trade-off to
make. If collection became expensive, the analysis to drop them is already done.

### "A competitor claims 95% accuracy."

Push back on the metric. "Accuracy" is a classification term and is meaningless for a continuous
target — ask 95% of *what*. Likely one of:

- **MAPE on high-value items only**, which flatters enormously (see the notebook's explanation
  of why MAPE was rejected: a $50 miss on a $33 item is a 150% error).
- **Training-set performance.** Our Decision Tree is "100% accurate" on training data and useless.
- **A different, easier target** — e.g. total store revenue rather than per-product-per-store.

**Our defensible claim:** RMSE $1,028 on data the model never saw, with a calibrated 80%
prediction interval whose coverage was measured at ~80%.

### "Deploy it tomorrow across all stores."

Recommend a staged rollout, and give reasons:

1. **Shadow mode first** — run alongside current planning for one cycle and compare. We have no
   measurement of BigMart's *current* process; the model is only proven against a naive average.
2. **Watch the tails.** Section 10.2 shows systematic under-prediction of top sellers. Those are
   the highest-revenue lines, so a blind rollout would under-stock exactly what matters most.
   Use the interval's upper bound for safety stock on those.
3. **Retraining trigger.** Data is a 2013 single-period snapshot with no seasonality. Establish
   monitoring on prediction error by segment before trusting it unattended.

### "Why not deep learning / XGBoost?"

- The brief requires scikit-learn only, so this is partly a constraint — say so, then give the
  real answer.
- On 8,523 rows and 9 features, gradient-boosted trees are the appropriate tool. Neural networks
  need far more data to beat trees on tabular problems, and would sacrifice the feature
  importance interpretability that makes this output *actionable*.
- The Decision Tree result shows the binding constraint is information in the features, not model
  capacity. A more powerful model would fit the same noise better, not find new signal.

---

## 3. Things to be able to explain on demand

| Concept | One-line answer |
|---|---|
| **Why a pipeline?** | Every cleaning rule refits per CV fold, so test data never influences training — and the deployed artifact contains the logic, so serving matches training exactly. |
| **What leakage did you avoid?** | Imputing before splitting. We measured it: worth ~$5 RMSE (0.46%) here — small, but the fix is free and the direction is always flattering. |
| **Why is `Non-Edible` derived from `Item_Type`?** | Because `Item_Identifier` doesn't exist at prediction time. We verified `Item_Type` determines the category exactly, so the rule is servable — otherwise the app would score products differently from how they were trained. |
| **Why RMSE not MAE for selection?** | Large misses cost disproportionately more in inventory. MAE is the better number to *communicate*, RMSE the better one to *optimise*. |
| **Why did you drop `Item_Category`?** | Tested it: CV difference was smaller than fold-to-fold noise. It's a strict grouping of `Item_Type`, so it adds columns but no information. |
| **How do the prediction intervals work?** | Two extra GB models trained with quantile loss at the 10th and 90th percentiles. No distributional assumption; verified ~80% empirical coverage on held-out data. |
| **Why did tuning barely help?** | It helped a small, *verified* amount — the paired CV confidence interval excludes zero. On the single split it was ~$11, which alone would have been within noise. Most performance comes from model family choice, not tuning. |
| **Why is the deployed model refit on 100% of data?** | Standard once selection is done — the test set already did its job. The app labels its metrics as held-out estimates, not measurements of the shipped object. |

---

## 4. The 3-minute pitch skeleton

1. **Problem (25s)** — BigMart plans inventory across 1,559 products × 10 outlets. Getting it
   wrong means stock-outs or dead capital. Today that decision leans on category averages.
2. **Solution (30s)** — live demo: change outlet type from Supermarket Type3 to Grocery Store,
   watch the forecast collapse. That is the single most persuasive interaction in the app.
3. **Evidence (45s)** — 45% of forecast error removed vs. the status quo. Four algorithms
   compared under repeated cross-validation. Every forecast ships with a calibrated 80% range.
4. **Insight (45s)** — the two levers that matter are *price band* and *store format*. Outlet
   format conversion moves more revenue than any individual product decision. That is a strategy
   finding, not just a model output.
5. **Honesty + ask (35s)** — name one limitation before being asked (tail bias, or new stores),
   then state the ask: a shadow-mode pilot for one planning cycle.

**Lead the demo with the Grocery Store toggle.** It is visual, instant, and makes the business
point better than any metric.
