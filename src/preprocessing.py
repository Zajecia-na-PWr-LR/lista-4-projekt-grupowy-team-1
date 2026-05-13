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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=RANDOM_STATE)
    salaries_train = X_train["EstimatedSalary"].values
    salaries_test = X_test["EstimatedSalary"].values
    return X_train, X_test, y_train, y_test, salaries_train ,salaries_test

def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])

def get_preprocessed_split(df, test_size=0.2):
    X_train, X_test, y_train, y_test, salaries_train, salaries_test = get_split(df, test_size)
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    return X_train_t, X_test_t, y_train, y_test, salaries_train, salaries_test, preprocessor