import numpy as np
import pandas as pd

class LogisticRegression:
	def __init__(self, X, y, nombres_features, L2 = 0):
		self.X = np.column_stack((np.ones(X.shape[0]), X))
		self.y = y
		self.nombres_features = nombres_features
		self.L2 = L2
	
	def entrenar_gradiente_descendiente(self,alpha,iters):
		self.w = np.zeros((self.X).shape[1])
		n = (self.X).shape[0]

		for _ in range (iters):
			z = self.X @ self.w
			y_pred = 1 / (1 + np.exp(-z))
			grad = (1/n)*self.X.T@(y_pred - self.y) + self.L2*self.w
			self.w = self.w - alpha*grad 
		
		return self.w
	
	def coefs_con_features(self):
		lista_noms = self.nombres_features.tolist()
		print(f'{round(self.w[0],4)} (bias)')
		for i in range((self.w).shape[0]):
			if i != 0:
				print(f' {round(self.w[i],4)} x {lista_noms[i-1]}')

	def predecir(self, X_val):
		X_val_bias = np.column_stack((np.ones(X_val.shape[0]), X_val))
		z = X_val_bias @ self.w
		return 1 / (1 + np.exp(-z))
	
	def predecir_clase(self, prediccion, umbral):
		
		return (prediccion >= umbral).astype(int)