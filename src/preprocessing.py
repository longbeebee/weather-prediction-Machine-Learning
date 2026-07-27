import pandas as pd

from .config import DROPPED_COLUMNS, PROCESSED_DATA_PATH


class Preprocessor:
    def filter_noon_records(self, df: pd.DataFrame) -> pd.DataFrame:
        noon = df[df["time"].dt.hour == 12].copy()
        return noon.reset_index(drop=True)

    def preprocess(self, df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
        df = self.filter_noon_records(df)
        df = df.drop(columns=[c for c in DROPPED_COLUMNS if c in df.columns])
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        if save:
            PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(PROCESSED_DATA_PATH, index=False)
        return df
