import numpy as np
import pandas as pd

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
