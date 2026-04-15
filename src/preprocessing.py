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

	