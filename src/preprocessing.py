import numpy as np
import pandas as pd

def reemplazo_NaNs(train,test,feature,columnas):
	'''
	La idea de utilizar medias_globales y por feature 
	es que las que son por feature, se agregara un valor
	mas preciso. En el caso de que se haga por escuela,
	el valor a agregar sera la media de cada columna pero
	basado en los valores de la escuela en particular.
	Para el test, si la escuela esta tambien en el train,
	se utiliza esa misma media. Si no se encuentra, se
	utiliza la media global de esa columna. De esta manera,
	los valores a agregar siempre son lo mas preciso posible.
	'''
	medias_globales = {}
	for f in columnas:
		train[f] = train[f].astype(float)
		test[f] = test[f].astype(float)
	
	for f in columnas:
		medias_por_cada_variable = train[f].mean()
		medias_globales[f] = medias_por_cada_variable

	medias_por_feature = {}
	for f in columnas:
		medias_por_cada_variable_y_feature = train.groupby(feature)[f].mean()
		medias_por_feature[f] = medias_por_cada_variable_y_feature

	for f in columnas:
		for esc in train[feature].unique():
			train.loc[(train[f].isna()) & (train[feature] == esc), f] = medias_por_feature[f][esc]
			if esc in test[feature].unique():
				test.loc[(test[f].isna()) & (test[feature] == esc), f] = medias_por_feature[f][esc]

	for f in columnas:
   		test.loc[test[f].isna(), f] = medias_globales[f]
	
	#modifica in place


def normalizar (train,test, columnas):
	train_normalizado = train.copy()
	media_std = {}
	for col in columnas:
			media_std[col] = (train_normalizado[col].mean(), train_normalizado[col].std())

	#Consigo media y stdianza para poder normalizar
	#--> Formula de normalizacion a usar = (x - media) / desvio
	test_normalizado = test.copy()
	for parte in [train_normalizado,test_normalizado]:
		for col in columnas:
				media,std = media_std[col]
				parte[col] = (parte[col] - media) / std
	
	return train_normalizado,test_normalizado,media_std
	
def desnormalizar (nombre_col, DataF_col_values,media_std):
	valores_normalizados = DataF_col_values.copy()
	media, std = media_std[nombre_col]
	valores_normalizados = valores_normalizados * std + media

	return valores_normalizados

def undersampling(X, y):
	clase_0_idx = np.where(y == 0)[0]
	clase_1_idx = np.where(y == 1)[0]
	
	if len(clase_0_idx) < len(clase_1_idx):
		minoritaria_idx = clase_0_idx
		mayoritaria_idx = clase_1_idx
	else:
		minoritaria_idx = clase_1_idx
		mayoritaria_idx = clase_0_idx

	mayoritaria_sample = np.random.choice(mayoritaria_idx, len(minoritaria_idx), replace=False)
	
	indices = np.concatenate([minoritaria_idx, mayoritaria_sample])
	np.random.shuffle(indices)
	
	return X[indices], y[indices]

def oversampling(X, y):
	clase_0_idx = np.where(y == 0)[0]
	clase_1_idx = np.where(y == 1)[0]
	
	if len(clase_0_idx) < len(clase_1_idx):
		minoritaria_idx = clase_0_idx
		n_mayoritaria = len(clase_1_idx)
	else:
		minoritaria_idx = clase_1_idx
		n_mayoritaria = len(clase_0_idx)
	
	duplicados_idx = np.random.choice(minoritaria_idx, n_mayoritaria, replace=True)
	
	indices = np.concatenate([np.arange(len(y)), duplicados_idx])
	np.random.shuffle(indices)
	
	return X[indices], y[indices]

def smote(X, y, k=5):
	'''
	crea muestras sinteticas de la clase minoritaria. Para cada ejemplo minoritario, busca sus K vecinos mas cercanos 
	y crea un nuevo ejemplo en un punto random entre el ejemplo y uno de sus vecinos. De esta manera
	genera datos nuevos (no duplicados) que ayudan al modelo a aprender mejor la clase minoritaria
	'''
	
	clase_0_idx = np.where(y == 0)[0]
	clase_1_idx = np.where(y == 1)[0]
	
	if len(clase_0_idx) < len(clase_1_idx):
		minoritaria_idx = clase_0_idx
		n_sinteticos = len(clase_1_idx) - len(clase_0_idx)
	else:
		minoritaria_idx = clase_1_idx
		n_sinteticos = len(clase_0_idx) - len(clase_1_idx)
	
	X_min = X[minoritaria_idx]
	sinteticos = []
	
	for _ in range(n_sinteticos):
		#elegir ejemplo aleatorio de la clase minoritaria
		idx = np.random.randint(len(X_min))
		ejemplo = X_min[idx]
		
		#calcula distancias euclidianas a todos los otros ejemplos minoritarios
		distancias = np.sqrt(((X_min - ejemplo) ** 2).sum(axis=1))
		distancias[idx] = np.inf  # ignorar el mismo ejemplo
		
		#elige uno de los k vecinos mas cercanos (random)
		vecinos_idx = np.argsort(distancias)[:k]
		vecino = X_min[np.random.choice(vecinos_idx)]
		
		#crear ejemplo sintetico en un punto random entre ejemplo y vecino
		alpha = np.random.random()
		nuevo = ejemplo + alpha * (vecino - ejemplo)
		sinteticos.append(nuevo)
	
	X_sinteticos = np.array(sinteticos)
	y_sinteticos = np.full(n_sinteticos, y[minoritaria_idx[0]])
	
	X_balanceado = np.vstack([X, X_sinteticos])
	y_balanceado = np.concatenate([y, y_sinteticos])
	
	indices = np.random.permutation(len(y_balanceado))
	return X_balanceado[indices], y_balanceado[indices]
	