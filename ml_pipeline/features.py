"""Feature definition for the contract classifier.

Pure functions: DataFrame in, DataFrame out (no database access here).
The ML module owns WHICH columns are features; the `ingestion` app owns what is stored.
"""
import pandas as pd

# The 23 numeric risk screens of the dataset (see docs/eda_findings.md).
FEATURE_COLUMNS = [
    "lot_bidscount", "lot_bidscount_avg", "lot_bidscount_avg_lag",
    "singleb_avg", "singleb_avg_lag",
    "bid_isconsortium", "bid_issubcontracted", "bid_issubcontracted_avg", "bid_issubcontracted_avg_lag",
    "relative_value", "relative_value_avg", "relative_value_avg_lag",
    "cpv3_avg_bidder_yearly", "cpv3_avg_bidder_yearly_avg_lag",
    "buyer_avg_bidder_yearly", "buyer_avg_bidder_yearly_avg_lag",
    "contract_count_bidder_yearly",
    "MAD_buyer", "benfords_buyer_yearly_avg", "benfords_buyer_yearly_avg_lag",
    "MAD_market", "benfords_market_yearly_avg", "benfords_market_yearly_avg_lag",
]

# Never features. tender_year: in this dataset it mostly reflects whether a contract
# comes from before or after a cartel was busted (a dataset-construction artifact).
EXCLUDED_COLUMNS = [
    "is_cartel", "cartel_id", "cartel_tender", "persistent_id", "tender_id", "lot_id",
    "bidder_id", "buyer_id", "tender_year", "country", "cpv2_smp",
]

assert not set(FEATURE_COLUMNS) & set(EXCLUDED_COLUMNS), "a feature is also listed as excluded"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model input (float columns, NaN allowed) from a frame holding the screens."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURE_COLUMNS].astype(float)
