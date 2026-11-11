import pandas as pd

from core.models import Contract


def load_frame() -> pd.DataFrame:
    """All contracts as one DataFrame: contract_id, cartel_id, is_cartel, tender_year, country + the screens."""
    rows = list(Contract.objects.values("id", "cartel_id", "is_cartel", "tender_year", "country", "screens"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"id": "contract_id"})
    screens = pd.DataFrame(df.pop("screens").tolist(), index=df.index)
    return pd.concat([df, screens], axis=1)
