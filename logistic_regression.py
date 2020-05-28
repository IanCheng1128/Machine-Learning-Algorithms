import numpy as np 


def load_data(path):
    dataList = []; labelList = []
    fr = open(path, 'r')
    for line in fr.readlines():
        curLine = line.strip().split(',')

        if int(curLine[0]) == 0:
            labelList.append(1)
        else:
            labelList.append(0)
        
        dataList.append([int(num)/255 for num in curLine[1:]])

    m = len(labelList)
    return np.array(dataList), np.array(labelList).reshape((m, 1))

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

class Logistic(object):
	def __init__(self, alpha):
		# learning rate
		self.alpha = alpha

	def initialize(self, X):
		self.w = np.zeros((X.shape[1], 1))
		self.b = 0

	def h_x(self, X, w, b):
		z = np.dot(X, w) + b
		return sigmoid(z)

	def grad(self, X, y):
		m = X.shape[0]
		hx = self.h_x(X, self.w, self.b)
		cost = (-np.dot(y.T, np.log(hx)) - np.dot((1-y).T, np.log(1-hx))) / m
		dw = np.dot(X.T, hx - y) / m
		db = np.sum(hx - y) / m
		grads = {'dw':dw, 'db':db}
		return grads, cost

	def fit(self, X, y, iter):
		self.initialize(X)
		self.cost = []
		for i in range(iter):
			grads, cost = self.grad(X, y)
			self.w -= grads['dw'] * self.alpha
			self.b -= grads['db'] * self.alpha

			if i%100 == 0:
				self.cost.append(cost)
				print(f"Cost after {i} iterations: ", cost)

	def predict(self, X):
		hx = self.h_x(X, self.w, self.b)
		y_pred = np.array([1 if x > 0.5 else 0 for x in list(hx)])
		return y_pred.reshape((X.shape[0], 1))

	def test(self, X, y):
		y_pred = self.predict(X)
		accurate_num = np.sum(y_pred == y)
		return accurate_num / y.shape[0]

if __name__ == '__main__':
	train_path = '/Users/iancheng/study/Coursera/ML/ML algorithm realization/Mnist/mnist_train.csv'
	test_path = '/Users/iancheng/study/Coursera/ML/ML algorithm realization/Mnist/mnist_test.csv'
	train_X, train_y = load_data(train_path)
	test_X, test_y = load_data(test_path)

	model = Logistic(0.01)
	model.fit(train_X, train_y, 1001)
	acc = model.test(test_X, test_y)

	print("Accuracy = ", acc)





