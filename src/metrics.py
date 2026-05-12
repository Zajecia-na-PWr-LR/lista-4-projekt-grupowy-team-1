import numpy as np

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