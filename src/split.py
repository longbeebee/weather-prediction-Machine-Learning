import pandas as pd


def chronological_split(df: pd.DataFrame):
    df = df.sort_values("time").reset_index(drop=True)
    years = set(df["time"].dt.year)
    if {2020, 2021, 2022, 2023, 2024, 2025}.issubset(years):
        train = df[df["time"].dt.year <= 2023]
        val = df[df["time"].dt.year == 2024]
        test = df[df["time"].dt.year == 2025]
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )
