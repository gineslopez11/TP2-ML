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
    
	if (TP + FP) == 0:
		return np.nan

	return TP / (TP + FP)

def recall(y_pred, y_real, clase_positiva=1):
	TP = ((y_pred == clase_positiva) & (y_real == clase_positiva)).sum()
	FN = ((y_pred == (1-clase_positiva)) & (y_real == clase_positiva)).sum()

	if (TP + FN) == 0:
		return np.nan

	return TP / (TP + FN)

def F1_score(y_pred, y_real, clase_positiva=1):
	p = precision(y_pred, y_real, clase_positiva)
	r = recall(y_pred, y_real, clase_positiva)
    
	return 2 * (p * r) / (p + r)

def curva_ROC(y_pred_prob, y_real, clase_positiva=1):
    umbrales = np.sort(np.unique(y_pred_prob))[::-1]
    tpr_list = []  # recall
    fpr_list = []  # tasa de FP
    
    for umbral in umbrales:
        y_pred = (y_pred_prob >= umbral).astype(int)
        if clase_positiva == 0:
            y_pred = 1 - y_pred
            y_real_bin = 1 - y_real
        else:
            y_real_bin = y_real
            
        TP = ((y_pred == 1) & (y_real_bin == 1)).sum()
        FP = ((y_pred == 1) & (y_real_bin == 0)).sum()
        FN = ((y_pred == 0) & (y_real_bin == 1)).sum()
        TN = ((y_pred == 0) & (y_real_bin == 0)).sum()
        
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), umbrales

def AUC_ROC(fpr, tpr):
    orden = np.argsort(fpr) #se necesitan valores en orden creciente sino da negativo el auc lo cual es imposible
    return np.trapz(tpr[orden], fpr[orden])

def curva_PR(y_pred_prob, y_real, clase_positiva=1):
    umbrales = np.sort(np.unique(y_pred_prob))[::-1]
    precision_list = []
    recall_list = []
    
    for umbral in umbrales:
        y_pred = (y_pred_prob >= umbral).astype(int)
        if clase_positiva == 0:
            y_pred = 1 - y_pred
            y_real_bin = 1 - y_real
        else:
            y_real_bin = y_real
            
        TP = ((y_pred == 1) & (y_real_bin == 1)).sum()
        FP = ((y_pred == 1) & (y_real_bin == 0)).sum()
        FN = ((y_pred == 0) & (y_real_bin == 1)).sum()
        
        prec = TP / (TP + FP) if (TP + FP) > 0 else 1
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        precision_list.append(prec)
        recall_list.append(rec)
    
    return np.array(recall_list), np.array(precision_list), umbrales

def AUC_PR(recall, precision):
    orden = np.argsort(recall)
    return np.trapz(precision[orden], recall[orden])