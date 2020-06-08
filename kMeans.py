import numpy as np 

class KMeans(object):
	def __init__(self, num_class, max_iter):
		self.num_class = num_class
		self.max_iter = max_iter


	def fit(self, X):
		m, n = X.shape
		# initialize the class points, distance, clusters
		centers = np.random.randn(self.num_class, n)
		distance = np.zeros((m, self.num_class))
		clusters = np.zeros((m, 1))

		for itr in range(self.max_iter):
			# calculate the distance to get the best class for each data
			for k in range(self.num_class):
				distance[:, k] = np.linalg.norm(X-centers[k], axis=1)

			# update the clusters
			clusters = np.argmin(distance, axis=1)


			# update the centers using mean values
			temp_centers = np.zeros((self.num_class, n))
			for k in range(self.num_class):
				data = X[clusters == k]
				centers[k] = np.mean(data, axis=0)

		self.centers = centers


	def predict(self, X):
		clusters = np.zeros((X.shape[0], 1))
		distance = np.zeros((X.shape[0], self.num_class))
		for k in range(self.num_class):
			distance[:, k] = np.linalg.norm(X-self.centers[k], axis=1)
		clusters = np.argmin(distance, axis=1)
		return clusters

if __name__ == '__main__':
	a = np.array([(3,2),(2,2),(1,2),(0,1),(1,0),(1,1),(5,6),(7,7),(9,10),(11,13),(12,12),(12,13),(13,13)])

	model = KMeans(2, 100)
	model.fit(a)
	res = model.predict(a)
	print(res)