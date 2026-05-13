import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.base import clone

def calculate_profit(y_true, y_pred, salaries):
    profit = 0
    for true, pred, salary in zip(y_true, y_pred, salaries):
        if true == 1 and pred == 1:    # TP poprawna znizka
            profit += 0.001 * salary
        elif true == 0 and pred == 0:  # TN  poprawny brak znizki
            profit += 0.003 * salary
        elif true == 0 and pred == 1:  # FP niepoprawna znizka 
            profit += 0.001 * salary
        elif true == 1 and pred == 0:  # FN niepoprawny brak znizki
            profit += 0
    return round(profit, 2)


def profit_summary(y_true, y_pred, salaries):
    model_profit = calculate_profit(y_true, y_pred, salaries)
    
    # dajemy znizke kazdemu (dla porownania)
    all_discount = calculate_profit(y_true, np.ones_like(y_true), salaries)
    
    # nikomu nie dajemy znizki (tez dla porownania)
    no_discount = calculate_profit(y_true, np.zeros_like(y_true), salaries)
    
    print(f"Nasz model:              {model_profit:>12.2f} $")
    print(f"Zniżka dla wszystkich:   {all_discount:>12.2f} $")
    print(f"Brak zniżek:             {no_discount:>12.2f} $")
    print(f"Przewaga nad all_discount: {model_profit - all_discount:>10.2f} $")
    print(f"Przewaga nad no_discount:  {model_profit - no_discount:>10.2f} $")
    
    return {
        "model": model_profit,
        "all_discount": all_discount,
        "no_discount": no_discount
    }
    
def tune_threshold(y_true, scores, salaries, n_thresholds=50):
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, n_thresholds)))
    best_t, best_p = thresholds[0], -np.inf
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        p = calculate_profit(y_true, y_pred, salaries)
        if p > best_p:
            best_p, best_t = p, t
    return best_t, best_p


def cv_with_threshold(model, X, y, salaries, n_splits=5, stratified=True, random_state=42):
    y = np.asarray(y)
    salaries = np.asarray(salaries)
    splitter = (StratifiedKFold if stratified else KFold)(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    profits, thresholds = [], []
    for tr, val in splitter.split(X, y):
        inner_tr, inner_val = train_test_split(
            tr, test_size=0.2, stratify=y[tr], random_state=random_state
        )
        m = clone(model)
        m.fit(X[inner_tr], y[inner_tr])
        inner_val_scores = m.predict_proba(X[inner_val])[:, 1]
        t, _ = tune_threshold(y[inner_val], inner_val_scores, salaries[inner_val])
        val_scores = m.predict_proba(X[val])[:, 1]
        y_pred_val = (val_scores >= t).astype(int)
        p = calculate_profit(y[val], y_pred_val, salaries[val])
        profits.append(p)
        thresholds.append(t)
    return {
        "profits": np.array(profits),
        "thresholds": np.array(thresholds),
        "mean": float(np.mean(profits)),
        "std": float(np.std(profits)),
    }