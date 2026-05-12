import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

DROP_COLS = ["RowNumber", "CustomerId", "Surname", "Complain"]
TARGET = "Exited"

CATEGORICAL = ["Geography", "Gender", "Card Type"]
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
           "EstimatedSalary", "Point Earned", "Satisfaction Score"]
BINARY = ["HasCrCard", "IsActiveMember"]

def load_data(path="../data/Customer-Churn-Records.csv"):
    df = pd.read_csv(path).drop(columns=DROP_COLS)
    return df

def get_split(df, test_size=0.2):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=RANDOM_STATE)

def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])