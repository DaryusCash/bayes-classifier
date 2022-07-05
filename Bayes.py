
import matplotlib.pyplot as plt
import math
import numpy as np
import DataProcess


#Calculates the bayes Clasifier
#Using a gaussian distribution assumtion
class Gaussian():
    def init():
        pass

    def data_tranform(self, data):
        new_data = {}

        for subject in range(len(data)):
            subject_list = []

            for pose in range(len(data[subject])):
                pose_list = []

                for i in range(len(data[subject][pose])):
                    
                    pose_list.append(data[subject][pose][i][0])
                
                pose_list = np.array(pose_list)
                subject_list.append(pose_list)
            
            subject_list = np.array(subject_list)
            new_data[subject] = subject_list
    
        return new_data

    #Gaussian
    def gaussian(self, mean, variance, x):
       
        scalar = (1 / (math.sqrt(2 * math.pi * variance)))
        return scalar * math.exp(-(math.pow(x - mean, 2) / (2 * variance)))
    

    #mean and variance for each feature
    def train(self, train_data, train_labels, num_dimensions):
        self.class_dict = {}
        
        pose = 0
        subject = 0 
        subject_stat = []

        prev_subject = 0
        pixel_avg = []

        train_data_dict = DataProcess.list_to_dict(train_data, train_labels) 

        for i in range(num_dimensions):
            
            for subject in range(len(train_data_dict)):

                for pose in range(len(train_data_dict[subject])):
                    pixel_avg.append(train_data_dict[subject][pose][i]) 


                new_pixel_avg = np.array(pixel_avg)
                stat = (np.mean(new_pixel_avg), np.var(new_pixel_avg))
            
                if subject not in self.class_dict.keys():
                    self.class_dict[subject] = []

                self.class_dict[subject].append(stat)
                    
                    
                pixel_avg = []
                    
    
    #liklihood for each paticular test data return class with highest liklihood
    def test(self, test_sample):
        liklihood_dict = {}
        liklihood = 0

        for subject in range(len(self.class_dict.keys())):
                
            for i in range(len(test_sample)):
                
                gauss = self.gaussian(self.class_dict[subject][i][0], self.class_dict[subject][i][1], test_sample[i])
                    
                if gauss > 0:
                    liklihood += math.log(gauss)
                        
            liklihood_dict[subject] = liklihood
            liklihood = 0
        
        return max(liklihood_dict, key=lambda k: liklihood_dict[k])

    #accuracy
    def eval(self, test_data, test_labels):
        total = 0
        score = 0
        subject =  0

        for pose in range(len(test_data)):

            if self.test(test_data[pose]) == test_labels[pose]:
                score += 1
            
            total += 1
                
        return score/total

