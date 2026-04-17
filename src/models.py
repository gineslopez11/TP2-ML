import numpy as np
import pandas as pd

class LogisticRegression:
	def __init__(self, X, y, nombres_features, L2=0, alpha=0.01, iters=1000, class_weight=None):
		self.X = np.column_stack((np.ones(X.shape[0]), X))
		self.y = y
		self.nombres_features = nombres_features
		self.L2 = L2
		self.alpha = alpha
		self.iters = iters
		self.class_weight = class_weight
	
	def entrenar_gradiente_descendiente(self):
		self.w = np.zeros((self.X).shape[1])
		n = (self.X).shape[0]

		for _ in range (self.iters):
			z = self.X @ self.w
			y_pred = 1 / (1 + np.exp(-z))
			error = y_pred - self.y
			if self.class_weight is not None:
				pesos = np.where(self.y == 0, self.class_weight[0], self.class_weight[1])
				error = error * pesos
			grad = (1/n)*self.X.T @ error + self.L2*self.w

			self.w = self.w - self.alpha*grad 
		
		return self.w
	
	def fit(self):
		self.entrenar_gradiente_descendiente()
	
	def coefs_con_features(self):
		lista_noms = self.nombres_features.tolist()
		print(f'{round(self.w[0],4)} (bias)')
		for i in range((self.w).shape[0]):
			if i != 0:
				print(f' {round(self.w[i],4)} x {lista_noms[i-1]}')

	def predecir_proba(self, X_val):
		X_val_bias = np.column_stack((np.ones(X_val.shape[0]), X_val))
		z = X_val_bias @ self.w
		return 1 / (1 + np.exp(-z))
	
	def predecir_clase(self, X, umbral=0.5):
		prediccion = self.predecir_proba(X)
		return (prediccion >= umbral).astype(int)

class LDA:
	def __init__(self, X, y, nombres_features = 0, L2=0, alfa=0, iters=0): #agrego mismos parametros que logistic para que cross val los acepte (duck typing)
		self.X = X
		self.y = y

	def fit(self):
		self.clases = np.unique(self.y)
		n, p = self.X.shape
		
		self.medias = {}
		self.pis = {}
		self.sigma = np.zeros((p, p))
		
		for k in self.clases:
			X_k = self.X[self.y == k]
			self.medias[k] = X_k.mean(axis=0)
			self.pis[k] = len(X_k) / n
			diff = X_k - self.medias[k]
			self.sigma += diff.T @ diff
		
		self.sigma /= n

	def predecir_proba(self, X):
		sigma_inv = np.linalg.pinv(self.sigma)
		scores = []
		for k in self.clases:
			mu_k = self.medias[k]
			pi_k = self.pis[k]
			score = X @ sigma_inv @ mu_k - 0.5 * mu_k @ sigma_inv @ mu_k + np.log(pi_k)
			scores.append(score)
		scores = np.array(scores).T
		# convierte scores a probabilidades con softmax
		scores_exp = np.exp(scores - scores.max(axis=1, keepdims=True))
		return scores_exp / scores_exp.sum(axis=1, keepdims=True)

	def predecir_clase(self, X, umbral = 0.5):
		sigma_inv = np.linalg.pinv(self.sigma)
		scores = []
		
		for k in self.clases:
			mu_k = self.medias[k]
			pi_k = self.pis[k]
			score = X @ sigma_inv @ mu_k - 0.5 * mu_k @ sigma_inv @ mu_k + np.log(pi_k)
			scores.append(score)
		
		scores = np.array(scores).T
		return self.clases[np.argmax(scores, axis=1)]

class LogisticRegressionMulticlase:
    def __init__(self, X, y, nombres_features, L2=0, alfa=0.01, iters=1000):
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.y = y
        self.nombres_features = nombres_features
        self.L2 = L2
        self.alfa = alfa
        self.iters = iters
        self.clases = np.unique(y)
        self.n_clases = len(self.clases)

    def _one_hot(self, y):
        n = len(y)
        Y = np.zeros((n, self.n_clases))
        for i, clase in enumerate(self.clases):
            Y[:, i] = (y == clase).astype(int)
        return Y

    def _softmax(self, Z):
        Z_exp = np.exp(Z - Z.max(axis=1, keepdims=True))  #estabilidad numérica
        return Z_exp / Z_exp.sum(axis=1, keepdims=True)

    def fit(self):
        n, p = self.X.shape
        self.W = np.zeros((p, self.n_clases))
        Y_onehot = self._one_hot(self.y)

        for _ in range(self.iters):
            Z = self.X @ self.W
            Y_pred = self._softmax(Z)
            grad = (1/n) * self.X.T @ (Y_pred - Y_onehot) + self.L2 * self.W
            self.W = self.W - self.alfa * grad

    def predecir_proba(self, X):
        X_bias = np.column_stack((np.ones(X.shape[0]), X))
        Z = X_bias @ self.W
        return self._softmax(Z)

    def predecir_clase(self, X, umbral=None):
        proba = self.predecir_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.clases[indices]

class Nodo:
	def __init__(self,feature=None, umbral=None, izquierda=None, derecha=None, clase=None, ig =None):
		self.feature = feature
		self.umbral = umbral
		self.izquierda = izquierda
		self.derecha = derecha
		self.clase = clase
		self.ig = ig

class ArbolDecision:		
	def __init__(self,X,y,max_profundidad, min_muestras_hoja):
		self.X = X
		self.y = y
		self.max_profundidad = max_profundidad
		self.min_muestras_hoja = min_muestras_hoja
	
	def _entropia(self, y): #es metodo solo para la clase (por el _ (convencion))
		clases, conteos = np.unique(y, return_counts=True)
		proporciones = conteos / len(y)
		return -np.sum(proporciones * np.log2(proporciones + 1e-10)) #se le suma eso para que no pueda dar 0
	
	def _information_gain(self, y, X_columna, umbral): #antes de dividir el noto tiene una "mezcla de clases". Con el IG podemos medir cuanto bajo esa mezcla (se vuelve mas pura si baja)
		H_padre = self._entropia(y)
		
		izq_idx = X_columna <= umbral
		der_idx = X_columna > umbral
		
		if izq_idx.sum() == 0 or der_idx.sum() == 0:
			return 0
		
		n = len(y)
		H_izq = self._entropia(y[izq_idx])
		H_der = self._entropia(y[der_idx])
		
		return H_padre - (izq_idx.sum()/n) * H_izq - (der_idx.sum()/n) * H_der
	
	def _mejor_division(self, X, y):
		mejor_ig = -1
		mejor_feature = None
		mejor_umbral = None
		
		for feature_idx in range(X.shape[1]):
			umbrales = np.unique(X[:, feature_idx])
			
			for umbral in umbrales:
				ig = self._information_gain(y, X[:, feature_idx], umbral)
				
				if ig > mejor_ig:
					mejor_ig = ig
					mejor_feature = feature_idx
					mejor_umbral = umbral
		
		return mejor_feature, mejor_umbral	
	
	def _construir(self, X, y, profundidad=0):
		n_muestras = len(y)
		if n_muestras == 0:
			return Nodo(clase=0)
		
		if (profundidad >= self.max_profundidad or n_muestras < self.min_muestras_hoja or len(np.unique(y)) == 1):
			clases_unicas, conteos = np.unique(y, return_counts=True)
			clase = clases_unicas[np.argmax(conteos)]
			return Nodo(clase=clase)
		
		mejor_feature, mejor_umbral = self._mejor_division(X, y)
		
		if mejor_feature is None:
			clases_unicas, conteos = np.unique(y, return_counts=True)
			clase = clases_unicas[np.argmax(conteos)]
			return Nodo(clase=clase)
		
		izq_idx = X[:, mejor_feature] <= mejor_umbral
		der_idx = X[:, mejor_feature] > mejor_umbral
		
		izquierda = self._construir(X[izq_idx], y[izq_idx], profundidad + 1)
		derecha = self._construir(X[der_idx], y[der_idx], profundidad + 1)
		
		return Nodo(feature=mejor_feature, umbral=mejor_umbral, izquierda=izquierda, derecha=derecha, ig=self._information_gain(y, X[:, mejor_feature], mejor_umbral))

	def fit(self):
		self.raiz = self._construir(self.X, self.y)

	def _predecir_uno(self, x, nodo):
		if nodo.clase is not None:
			return nodo.clase
		if x[nodo.feature] <= nodo.umbral:
			return self._predecir_uno(x, nodo.izquierda)
		else:
			return self._predecir_uno(x, nodo.derecha)

	def predecir_clase(self, X, umbral=None):
		return np.array([self._predecir_uno(x, self.raiz) for x in X])
	
	def _importancia_nodos(self, nodo, importancias):
		if nodo.clase is not None:
			return
		importancias[nodo.feature] += nodo.ig
		self._importancia_nodos(nodo.izquierda, importancias)
		self._importancia_nodos(nodo.derecha, importancias)

	def feature_importances(self, n_features):
		importancias = np.zeros(n_features)
		self._importancia_nodos(self.raiz, importancias)
		total = importancias.sum()
		return importancias / total if total > 0 else importancias
	
class RandomForest:
    def __init__(self, X, y, nombres_features, L2=0, alfa=0, iters=0, 
                 n_arboles=50, max_profundidad=5, min_muestras_hoja=10, max_features=None):
        self.X = X
        self.y = y
        self.nombres_features = nombres_features
        self.n_arboles = n_arboles
        self.max_profundidad = max_profundidad
        self.min_muestras_hoja = min_muestras_hoja
        self.max_features = max_features if max_features else int(np.sqrt(X.shape[1]))
        self.arboles = []

    def fit(self):
        n = len(self.y)
        for _ in range(self.n_arboles):
            indices = np.random.choice(n, n, replace=True)
            X_boot = self.X[indices]
            y_boot = self.y[indices]
            
            features_idx = np.random.choice(self.X.shape[1], self.max_features, replace=False)
            
            arbol = ArbolDecision(X_boot[:, features_idx], y_boot, self.max_profundidad, self.min_muestras_hoja)
            arbol.fit()
            self.arboles.append((arbol, features_idx))

    def predecir_clase(self, X, umbral=None):
        predicciones = np.array([arbol.predecir_clase(X[:, features_idx]) for arbol, features_idx in self.arboles])
        
        resultado = []
        for i in range(X.shape[0]):
            votos = predicciones[:, i]
            clases, conteos = np.unique(votos, return_counts=True)
            resultado.append(clases[np.argmax(conteos)])
        
        return np.array(resultado)

    def predecir_proba(self, X):
        predicciones = np.array([arbol.predecir_clase(X[:, features_idx]) for arbol, features_idx in self.arboles])
        
        clases_unicas = np.unique(self.y)
        probas = np.zeros((X.shape[0], len(clases_unicas)))
        
        for i in range(X.shape[0]):
            votos = predicciones[:, i]
            for j, clase in enumerate(clases_unicas):
                probas[i, j] = (votos == clase).sum() / len(votos)
        
        return probas
	
    def feature_importances(self):
        n_features = self.X.shape[1]
        importancias = np.zeros(n_features)
        for arbol, features_idx in self.arboles:
           imp_arbol = arbol.feature_importances(len(features_idx))
           importancias[features_idx] += imp_arbol
        return importancias / importancias.sum()