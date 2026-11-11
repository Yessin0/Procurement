# EDA & ML Baseline Findings

Working notes for the report. Source: `GTI_labelled_cartel_data_NOV2023.csv` (Mendeley Data, v2) and `notebooks/ml_baseline.ipynb`. Update this file as results change.

---

## 1. Dataset (see README Section 2 for the full table)

- 15,616 rows × 34 columns; 10,565 tenders; 1,001 suppliers; 2,984 buyers; 7 anonymized countries; 2004–2021; 73 cartel cases (`cartel_id`, never missing).
- **Winners only** (losing bidders absent) → no co-bidding graph; the network module uses supplier–buyer relationships.
- Every firm belongs to a known cartel. `is_cartel` = 1 for 41.9% of rows (a curated sample, not real prevalence). Random-model PR-AUC ≈ 0.42.
- 23 of the 73 cases contain a single label only; 50 contain both.
- Missing: `lot_id` 94%, `persistent_id` 77%, `buyer_id` 19%. No contract amount (only `relative_value`).
- Label construction (methodology paper): contracts of cartel firms *during* the cartel period are positive; the same firms' contracts *after* the cartel was busted are negative.

## 2. ML baseline protocol

- **Features:** the 23 numeric screens (no IDs, no `cartel_id`/`cartel_tender`, no `bidder_id`/`buyer_id`, **no `tender_year`**).
- **Validation:** `GroupKFold` (5 folds) by `cartel_id`; metrics: PR-AUC (main), ROC-AUC, precision/recall/F1 at 0.5.
- **Models:** dummy (prior), logistic regression, random forest, XGBoost. No imbalance handling needed (42% positives).
- Additional tests: random vs group split, leave-largest-case-out, leave-one-country-out, per-case view, per-year view.

## 3. Results (group split by cartel, 23 screens, no year)

| Model | PR-AUC (mean ± std over folds) | Random baseline |
|---|---|---|
| Dummy (prior) | 0.419 | 0.419 |
| Logistic regression | 0.501 ± 0.235 | 0.419 |
| Random forest | 0.610 ± 0.148 | 0.419 |
| XGBoost | 0.601 ± 0.140 | 0.419 |

- Differences of ~0.02 (e.g. adding sector/country: 0.628) are within fold-to-fold noise; folds range from 0.46 to 0.83 PR-AUC for the random forest.
- Fold 3 is below its baseline (PR-AUC 0.462 vs 0.520; ROC-AUC 0.415). It contains case `country_2_7` (1,899 of its 3,123 rows), where the model ranks contracts backwards (within-case ROC-AUC 0.356).
- Strongest single screens (case-weighted): `lot_bidscount_avg`, `lot_bidscount_avg_lag`, `benfords_market_yearly_avg`, `MAD_market`. Individual screens are weak once every case counts equally (0.50–0.58).
- Most important features in the XGBoost model: `lot_bidscount_avg`, `benfords_market_yearly_avg` (+ lag), `lot_bidscount_avg_lag`, `buyer_avg_bidder_yearly_avg_lag`.

## 4. Key finding: the label is strongly tied to time

- Share of positive contracts by year: 55–88% for 2004–2014, 36% in 2015, 9% in 2016, 2% in 2017, **0% from 2018 on**.
- In all 50 cases with both labels, negative contracts are later than positive ones on average.
- A model using only `tender_year` reaches PR-AUC **0.744 ± 0.214**. Random forest on screens + year reaches 0.768 (XGBoost 0.746); screens add almost nothing on top of the year.
- Leaving the largest case out: 1.000 PR-AUC / 0.999 ROC-AUC with the year, versus 0.871 / 0.704 without it.
- **Decision:** `tender_year` is excluded from the model. A real deployment would not know when a cartel ended, so the year reflects how the dataset was built, not collusion. The year-only result is reported as an artifact reference, not as a result.

## 5. Do the screens carry signal beyond time?

- Within-year ROC-AUC of the screens model (13 years with both labels and n ≥ 50): per-year values 0.38–0.78, size-weighted average about **0.65**. The two weakest years are 2005 (0.41) and 2006 (0.38), the earliest and smallest.
- A random forest predicts `tender_year` from the screens with R² = 0.171 (group split): the screens encode little time.
- **Reading:** the screens carry modest signal that is not just time. This is what they are designed for (behaviour changes when collusion stops), but the effect is weak.

## 6. Generalization limits

- **Random vs group split:** PR-AUC 0.988 (accuracy 94.8%) vs 0.610 (accuracy 66.0%). Many screens are buyer-year or bidder-year averages, so contracts of one firm share identical values and a random split lets the model recognise the firm. Comparable in size to the gap in the methodology paper.
- **Leave one country out:** mean accuracy 0.611 (paper: 54.7–67.4%), mean PR-AUC 0.444, barely above baseline. ROC-AUC ≈ 0.5 for `country_2`, `country_4`, `country_6`, `country_7`. The screens barely transfer across countries.
- **Per case (within-case ROC-AUC):** good in `country_5_11` (0.977), `country_3_5` (0.949), `country_2_12` (0.892), `country_2_17` (0.845); at or below chance in `country_7_1` (0.331), `country_2_7` (0.356), `country_4_10` (0.471), `country_1_5` (0.501). The average pattern does not hold for every cartel.

## 7. Decisions and open items

**Decided**
- Model: XGBoost on the 23 screens (tied with random forest within noise; smaller model file, fast SHAP).
- Operating point: flag the riskiest 10% of contracts (`--flag-rate`). The first version picked the F1-optimal threshold, which degenerated into flagging 99% of contracts (threshold 0.002, precision 0.42, recall 0.995): with 42% positives and a modest model, F1 is highest when almost everything is flagged. Report precision and lift at the chosen flag rate instead.
- Report evaluation protocol: group split by `cartel_id` (mean ± spread), within-year ROC-AUC, per-case table, leave-country-out, random-vs-group gap, year-only artifact reference.
- Wording for the report: the model ranks contracts by resemblance to collusion-period contracts of known cartels; it is a decision-support ranking, not a verdict.

**To check**
- `bid_issubcontracted` has means of 2.2 vs 3.3 although described as a flag — confirm its definition in the paper.
- `relative_value` definition (probably final price / estimate) — confirm in the paper.
- `contract_count_bidder_yearly` means (40.5 cartel vs 110.8 other) may be inflated by the single dominant supplier (1,973 rows).

**Next experiments (time-box them)**
- Own features from winners-only data (supplier share of a buyer's yearly contracts, buyers per supplier-year, buyer concentration HHI); keep them only if they improve the group-split and within-year scores.
- Precision at top-k (e.g. top 5% / 10% of contracts) as the operational metric, and threshold selection on out-of-fold predictions.
- Repeat the group split with several fold assignments to see how stable the 0.61 is.
