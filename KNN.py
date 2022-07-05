# !/usr/bin/env python


from queue import PriorityQueue
import math

import numpy as np


class KNNClassifier():
	def __init__(self, train_set, train_labels, neighbors=1):
		self.neighbors = neighbors
		self.train_set = train_set
		self.train_labels = train_labels
	
	 

		
    # # calculate the Euclidean distance between two vectors
	def euclidean_distance(self, row1, row2):
		distance = 0.0
		for i in range(len(row1)-1):
			distance += (row1[i] - row2[i])**2
		
		return math.sqrt(distance)

	
	# # Locate the most similar neighbors
	def get_neighbors(self, train, train_labels, test_row, num_neighbors):
		
		distances = PriorityQueue()

		for i in range(len(train)):
			dist = self.euclidean_distance(test_row, train[i])
			distances.put((dist, train_labels[i]))

		neighbors = []

		for i in range(num_neighbors):
			neighbors.append(distances.get())

		return neighbors

	def vote(self, neighbors):
		vote_dict = {}

		for i in range(len(neighbors)):
			
			label = neighbors[i][-1]

			if neighbors[i][1] not in vote_dict.keys():
				vote_dict[label] = 0

			vote_dict[label] += 1
		
		return sorted(vote_dict.items(), key=lambda k_v: k_v[1])[-1][0]

    # # Make a classification prediction with neighbors
	def predict(self, test_image):   
		neighbors = self.get_neighbors(self.train_set, self.train_labels, test_image, self.neighbors)

		return self.vote(neighbors)
    
	def eval(self, test_data, test_labels):
		correct = 0

		for i in range(len(test_data)):

			if self.predict(test_data[i]) == test_labels[i]:
				correct += 1

		return correct / len(test_data)

