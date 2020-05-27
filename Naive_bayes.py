import numpy as np

def load_data(path):
	X = []
	y = []
	with open(path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip().split(',')
			y.append(int(line[0]))
			X.append([int(int(x) > 128) for x in line[1:]])
	return np.array(X), np.array(y)

class Naive_bayes(object):

	def fit(self, X, y):
		'''
		return the estimated parameters
		'''
		# number of classes
		class_num = len(list(set(y)))
		# dimensions of data
		dim = X.shape[1]
		# Length of the dataset
		m = X.shape[0]

		# initialize the prior parameters
		phi_y = np.zeros((class_num, 1))

		# initialize the models parameters
		phi_X = np.zeros((class_num, dim, 2)) # bernulli distribution, 1/0

		# First, we need to build models for every class
		for i in range(class_num):
			phi_y[i] = (np.sum(y == i) + 1) / (m + class_num)


		# Secondly, we need to build models for every class and every feature
		for i in range(m):
			label = y[i]
			data = X[i]
			for j in range(len(data)):
				phi_X[label][j][data[j]] += 1

		for i in range(class_num):
			for j in range(dim):
				X_0 = phi_X[i][j][0]
				X_1 = phi_X[i][j][1]
				phi_X[i][j][0] = (X_0 + 1) / (X_0 + X_1 + 2)
				phi_X[i][j][1] = (X_1 + 1) / (X_0 + X_1 + 2)

		self.phi_y = phi_y
		self.phi_X = phi_X

	def predict(self, X):
		class_num = self.phi_y.shape[0]
		res = np.ones((len(X), class_num))
		for i in range(class_num):
			sum = np.ones(len(X))
			for j in range(X.shape[1]):
				sum = sum * self.phi_X[i][j][X[:,j]]
			res[:,i] = sum * self.phi_y[i]
		return res.argmax(axis=1)

	def test(self, X, y):
		predict_y = self.predict(X)
		accuract_num = 0
		for i in range(len(y)):
			if predict_y[i] == y[i]:
				accuract_num += 1
		return accuract_num / len(y)


if __name__ == '__main__':
	train_path = '/Users/iancheng/study/Coursera/ML/ML algorithm realization/Mnist/mnist_train.csv'
	test_path = '/Users/iancheng/study/Coursera/ML/ML algorithm realization/Mnist/mnist_test.csv'
	train_X, train_y = load_data(train_path)
	test_X, test_y = load_data(test_path)

	model = Naive_bayes()
	model.fit(train_X, train_y)
	accuracy = model.test(test_X, test_y)
	print("Accuracy = ", accuracy)


