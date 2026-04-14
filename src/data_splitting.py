import numpy as np
import pandas as pd
from src.models import LogisticRegression
from src.metrics import F1_score
from src.preprocessing import *

def random_split(dev,percentage,rand_state = None):
	dev_insuficiente = dev[dev['rendimiento_binario'] == 0]
	dev_suficiente = dev[dev['rendimiento_binario'] == 1]
	dev_total = [dev_insuficiente,dev_suficiente]
	trains = []
	tests = []

	for d in dev_total:

		d_shuffled = d.sample(frac = 1, random_state = rand_state) 
		total_filas = d_shuffled.shape[0]
		q_filas_train = int(percentage*total_filas)
		train_set_d = d_shuffled[:q_filas_train]
		test_set_d = d_shuffled[q_filas_train:]
		trains.append(train_set_d)
		tests.append(test_set_d)
	

	train_set = pd.concat([trains[0], trains[1]])
	test_set = pd.concat([tests[0], tests[1]])

	return train_set, test_set

def group_split(dev,m,rand_state = None):
	if rand_state != None:
		np.random.seed(rand_state)
	colegios = dev['escuela'].unique()
	indices = np.random.permutation(len(colegios)) #el orden de los colegios es random, no utiliza siempre el mismo orden
	seleccionados = colegios[indices[:m]]
	restantes = colegios[indices[m:]]
	train = dev[dev['escuela'].isin(seleccionados)]
	test = dev[dev['escuela'].isin(restantes)]

	return train, test

def temporal_split(dev,m):
	semestres = sorted(dev['semestre'].unique(), key=lambda x: (x.split('-')[0], x.split('-')[1]))
	seleccionados = semestres[:m]
	restantes = semestres[m:]
	train = dev[dev['semestre'].isin(seleccionados)]
	test = dev[dev['semestre'].isin(restantes)]

	return train, test

def cross_val(dev, nombres_features, y_col, K, L2, alfa, iters, umbral, group_key, columnas_continuas, tipo,clase_positiva):
	n = len(dev)  
	
	if tipo == 'aleatorio':
		indices = np.arange(n)
		np.random.shuffle(indices)
		folds = np.array_split(indices, K)

	elif tipo == 'group':
		grupos = dev[group_key].unique()
		folds = [np.where(dev[group_key] == g)[0] for g in grupos]

	elif tipo == 'temporal':
		semestres = sorted(dev['semestre'].unique(), key=lambda x: (x.split('-')[0], x.split('-')[1]))
		folds = [np.where(dev['semestre'] == s)[0] for s in semestres]

	F1s = []

	for i in range(len(folds)):
		val_idx = folds[i]
		train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])

		train_fold = dev.iloc[train_idx].copy()
		val_fold = dev.iloc[val_idx].copy()

		reemplazo_NaNs(train_fold, val_fold, group_key, columnas_continuas)
		train_fold_norm, val_fold_norm, _ = normalizar(train_fold, val_fold, columnas_continuas)

		X_train_fold = train_fold_norm[nombres_features].values
		y_train_fold = train_fold_norm[y_col].values
		X_val_fold = val_fold_norm[nombres_features].values
		y_val_fold = val_fold_norm[y_col].values

		modelo_train = LogisticRegression(X_train_fold, y_train_fold, nombres_features, L2)
		modelo_train.entrenar_gradiente_descendiente(alfa, iters)

		y_pred_prob = modelo_train.predecir(X_val_fold)
		y_pred_clase = modelo_train.predecir_clase(y_pred_prob, umbral)
		F1_i = F1_score(y_pred_clase, y_val_fold, clase_positiva)

		F1s.append(F1_i)

	return np.mean(F1s)  