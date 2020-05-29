import numpy as np 

def load_data(path):
	'''
	Load data 
	Input: file path
	Return: X, y, np.array
	'''
	X = []
	y = []
	with open(path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip().split(',')
			y.append(int(line[0]))
			X.append([int(int(x) > 128) for x in line[1:]])
	return np.array(X), np.array(y)


class KNN(object):
	def __init__(self, num_neighbors):
		# initialize the numbers of neighbors
		self.num_neighbors = num_neighbors


	def distance(self, x1, x2):
		# calculate the distance between two data points
		dis_square = np.square(x1 - x2)
		return np.sqrt(np.sum(dis_square))

	def neighbors(self, train_X, x):
		'''
		Get the K nearest neighbors of one point 
		Return: Index of K nearest neighbors
		'''
		m = train_X.shape[0]
		neighbor_dist = np.zeros(m)

		for i in range(m):
			distance = self.distance(x, train_X[i])
			neighbor_dist[i] = distance

		K_neighbors = np.argsort(neighbor_dist)[:self.num_neighbors]
		return K_neighbors

	def predict(self, train_X, train_y, X):
		'''
		Predict the labels of test dataset
		Return: labels, np.array, dim: length of X * 1
		'''
		m = X.shape[0]
		num_labels = len(list(set(train_y)))
		y_predict = np.zeros((m, 1))

		for i in range(m):
			neighbors = self.neighbors(train_X, X[i])
			negih_labels = train_y[neighbors]
			target_labels = np.zeros((num_labels, 1))

			for j in range(len(neighbors)):
				target_labels[int(train_y[neighbors[j]])] += 1
			
			y_predict[i] = target_labels.argmax()
		return y_predict

	def test(self, train_X, train_y, X, y):
		y_predict = self.predict(train_X, train_y, X)

		acc = np.sum(y_predict == y) / y.shape[0]
		return acc 

if __name__ == '__main__':
	train_path = '/Users/iancheng/study/Coursera/ML/ML algorithm realization/Mnist/mnist_train.csv'
	test_path = '/Users/iancheng/study/Coursera/ML/ML algorithm realization/Mnist/mnist_test.csv'
	train_X, train_y = load_data(train_path)
	test_X, test_y = load_data(test_path)

	K_neighbors = 30
	model = KNN(K_neighbors)
	acc = model.test(train_X, train_y, test_X, test_y)
	print("Accuracy = ", acc)