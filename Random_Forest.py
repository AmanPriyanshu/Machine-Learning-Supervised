from data_reader import main
import numpy as np

X, Y = main('./Data/abalone_C1_P02_V01_CA0.csv')

def gini_impurity(X, Y, nodes):
	gi = 0
	for n in range(len(nodes)):
		A = [0 if i<=nodes[n] else 1 for i in X.T[n]]
		p_i = np.sum(np.absolute(A-Y))/Y.shape[0]
		g_i = p_i*(1-p_i)
		gi += g_i
	return gi

def random_forest(X, Y, depth, ):
	print(X)

