import numpy as np

def matriz_confusion(y_pred, y_real, clase_positiva=1):
    TP = ((y_pred == clase_positiva) & (y_real == clase_positiva)).sum()
    TN = ((y_pred == (1-clase_positiva)) & (y_real == (1-clase_positiva))).sum()
    FP = ((y_pred == clase_positiva) & (y_real == (1-clase_positiva))).sum()
    FN = ((y_pred == (1-clase_positiva)) & (y_real == clase_positiva)).sum()
    
    return np.array([[TN, FP], [FN, TP]])

def accuracy(y_pred, y_real, clase_positiva=1):
    TP = ((y_pred == clase_positiva) & (y_real == clase_positiva)).sum()
    TN = ((y_pred == (1-clase_positiva)) & (y_real == (1-clase_positiva))).sum()
    FP = ((y_pred == clase_positiva) & (y_real == (1-clase_positiva))).sum()
    FN = ((y_pred == (1-clase_positiva)) & (y_real == clase_positiva)).sum()
    
    return (TP + TN) / (TP + TN + FP + FN)

def precision(y_pred, y_real, clase_positiva=1):
    TP = ((y_pred == clase_positiva) & (y_real == clase_positiva)).sum()
    FP = ((y_pred == clase_positiva) & (y_real == (1-clase_positiva))).sum()
    
    return TP / (TP + FP)

def recall(y_pred, y_real, clase_positiva=1):
    TP = ((y_pred == clase_positiva) & (y_real == clase_positiva)).sum()
    FN = ((y_pred == (1-clase_positiva)) & (y_real == clase_positiva)).sum()
    
    return TP / (TP + FN)

def F1_score(y_pred, y_real, clase_positiva=1):
    p = precision(y_pred, y_real, clase_positiva)
    r = recall(y_pred, y_real, clase_positiva)
    
    return 2 * (p * r) / (p + r)