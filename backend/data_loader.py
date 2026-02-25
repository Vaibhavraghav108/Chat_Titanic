
import os
import pandas as pd

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
data_path = os.path.join(data_dir, "titanic.csv")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
_df = None

def load_titanic_data():
    global _df
    if _df is not None:
        return _df
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(data_path):
        print(f"Downloading Titanic dataset...")
        df = pd.read_csv(url)
        df.to_csv(data_path, index=False)
        print(f"Saved to {data_path}")

    df = pd.read_csv(data_path)
    df = handle_missing_values(df)

    _df = df
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    return _df


def handle_missing_values(df, strategy="fill"):
    """
    Handle missing values in the dataframe.
    strategy: 'fill' -> fill missing values with median/mode
              'drop' -> drop rows with missing values
    """
    if strategy == "drop":
        df = df.dropna()
    else:
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    print(f"Missing values handled (strategy: {strategy}) — remaining nulls: {df.isnull().sum().sum()}")
    return df


def get_schema_info():
    df = load_titanic_data()
    lines = [f"Titanic Dataset - {len(df)} rows, {len(df.columns)} columns", "", "Columns:"]

    for col in df.columns:
        if df[col].dtype == "object":
            lines.append(f"  - {col} (text, {df[col].nunique()} unique)")
        else:
            lines.append(f"  - {col} (numeric, range: {df[col].min()}-{df[col].max()})")

    return "\n".join(lines)

