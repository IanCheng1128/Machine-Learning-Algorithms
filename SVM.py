import numpy as np 

def load_data(path):
    dataArr = []; labelArr = []
    fr = open(path)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        dataArr.append([int(num) / 255 for num in curLine[1:]])
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    m = len(labelArr)
    return np.array(dataArr), np.array(labelArr).reshape((m, 1))

class SVM(object):
	def __init__(self, X, y, C, tolerance, iterations, KTup):
		'''
		X: features
		y: lables
		C: penalty
		tolerance: soft margin
		iterations: iteration number
		KTup: list, 0,kernel type, 'lin' or 'rbf', 1,sigma
		'''
		self.X = X
		self.y = y

		# Hyperparameter
		self.C = C
		self.tolerance = tolerance
		self.iterations = iterations

		# Initialize parameters
		self.m, self.n = X.shape
		self.alpha = np.zeros((self.m, 1))
		self.kernel = self.cal_kernel(KTup)
		self.b = 0
		# self.support_vector = []

	def cal_kernel(self, KTup):
		'''
		Calculate the kernel function
		KTup: list, 0,kernel type, 'lin' or 'rbf', 1,sigma
		'''
		# kernel function, dim: m * m
		k = np.zeros((self.m, self.m))
		# Linear kernel
		if KTup[0] == 'lin':
			k = np.dot(self.X, self.X.T)

		# Gaussian kernel
		elif KTup[0] == 'rbf':
			for i in range(self.m):
				x = self.X[i]
				for j in range(i, self.m):
					z = self.X[j]
					res = np.exp(np.dot((x-z).T, x-z) / (-1 * KTup[1]**2))
					k[i][j] = res
					k[j][i] = res
		return k


	def cal_g(self, i):
		'''
		Calculate the predicted class based on the w, b, and alpha with index i
		w = sum(alpha_i * y_i * x)
		g(x) = sum(alpha_i * y_i * <x_i, x> + b)
		'''
		return np.dot((self.alpha * self.y).T, self.kernel[:,i]) + self.b

	def cal_E(self, i):
		'''
		Calculate the error of ith sample
		'''
		return self.cal_g(i) - self.y[i]

	def satisfy_KKT(self, i):
		g = self.cal_g(i)
		label = self.y[i]
		y_g = g * label

		if self.alpha[i] == 0:
			return y_g >= 1
		elif self.alpha[i]  == self.C:
			return y_g <= 1
		elif self.alpha[i] >0 and self.alpha[i] < self.C:
			return y_g == 1
		else:
			return False

	def select_rand(i):
		j = i
		while j == i:
			j = int(np.uniform(0, self.m))
		return j


	def train(self):
		itr = 0
		parameter_change = 1;

		# if itr reaches the max iterations or parameters don't change, then we stop
		while itr < self.iterations and parameter_change > 0:
			print(f"The {itr} iteration...")
			itr += 1
			parameter_change = 0

			# Pick the first alpha
			for i in range(self.m):
				if not self.satisfy_KKT(i):
					E1 = self.cal_E(i)

					# Random select the second alpha
					j = self.select_rand(i, self.m)
					E2 = self.cal_E(j)

					alpha_old1 = self.alpha[i].copy()
					alpha_old2 = self.alpha[j].copy()

					label1 = self.y[i]
					label2 = self.y[j]

					# satisfy the constraints
					if label1 != label2:
						L = max(0, alpha_old2 - alpha_old1)
						H = min(0, self.C, self.C + alpha_old2 - alpha_old1)
					else:
						L = max(0, alpha_old2 + alpha_old1 - self.C)
						H = min(0, self.C, alpha_old2 + alpha_old1)

					if L == H:
						print("L = H")
						continue

					#  Update alpha2
					# eta = K11 + K22 - 2*K12
					eta = self.kernel[i][i] + self.kernel[j][j] - 2 * self.kernel[i][j]
					alpha_new2 = alpha_old2 + label2 * (E1 - E2) / eta
					if alpha_new2 > H:
						alpha_new2 = H
					elif alpha_new2 < L:
						alpha_new2 = L
					
					# Update alpha1
					alpha_new1 = alpha_old1 + label1 * label2 * (alpha_old2 - alpha_new2)

					if abs(alpha_new2 - alpha_old2) >= 0.00001:
						parameter_change += 1
					else:
						print("alpha 2 not moving enough")
						continue

					self.alpha[i] = alpha_new1
					self.alpha[j] = alpha_new2


					# Update b
					b1 = -1 * E1 - label1 * self.kernel[i][i] * (alpha_new1 - alpha_old1) - label2 * self.kernel[j][i] * (alpha_new2 - alpha_old2) + self.b
					b2 = -1 * E2 - label1 * self.kernel[i][j] * (alpha_new1 - alpha_old1) - label2 * self.kernel[j][j] * (alpha_new2 - alpha_old2) + self.b

					if (alpha_new1>0 and alpha_new1<self.C):
						b_new = b1
					elif (alpha_new2>0 and alpha_new2<self.C):
						b_new = b2
					else:
						b_new = (b1 + b2) / 2.

					self.b = b_new


	def predict(self, X):
		pred = np.zeros(X.shape[0])
		for i in range(X.shape[0]):
			pred[i] = np.sign(self.cal_g(i))
		return pred

	def test(self, X, y):
		y_pred = self.predict(X)
		acc = 0
		for i in range(len(y)):
			if y_pred[i] == y[i]:
				acc += 1

		return acc / len(y)

if __name__ == '__main__':
	train_path = '/Users/iancheng/study/Coursera/ML/ML algorithm realization/Mnist/mnist_train.csv'
	test_path = '/Users/iancheng/study/Coursera/ML/ML algorithm realization/Mnist/mnist_test.csv'
	# train_X, train_y = load_data(train_path)
	test_X, test_y = load_data(test_path)

	model = SVM(test_X, test_y, 100, 0.001, 100, ['rbf', 10])
	model.train()
	accuracy = model.test(test_X, test_y)
	print("Accuracy = ", accuracy)















