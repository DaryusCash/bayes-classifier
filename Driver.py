# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Project 1
#    April 13, 2022
#    ENEE 436
#    Daryus Cash

# %% [markdown]
# Project Discription:
# In the following documentation, we will explore concepts of machine learning and classification algorithms. 
# We will discuss in particular the Bayes classifier and K-Nearest Neighbor classifier. We will also look at the dimensionality-reduction techniques such as PCA and LDA. We will see how these can be used to identify images and we will discuss the results of my findings. 

# %% [markdown]
# ##### Data Processing
# Before we can run the data through a classifier we must process the data to reduce the number of dimentions. In some case you have to process the data set because the size of the data is to large to be processed in a resonable amount of time. In our case we are processing the data to make the classifiers more effective in evaluating the data. 
#
# In task one, we were given 68 subjects with 21 images each. In task two, we had 200 subjects with three images each. In task two, each subject had a neutral poss, expression pose, and illumination(Light varying) pose. These data sets were given in image sizes of 24X21 pixels. To properly implement the classifiers we must first reduce the dimensionality of the data to a manageable size. We also can flatten the date from 2-D to 1-D which will allow our program to run in O(n) time vs O(nm) time. This will improve programs' efficiency and also make processing easier to code. Before we center and reduce the number of dimensions we had to split the data into two sets, Training and test data. We did this in a sudo-random fashion to give us some level of variability in our testing. 

# %% [markdown]
# ## Program Driver
# The following code segments are drivers for task one and task two. 

# %%
import KNN
import Bayes
import PCA
import LDA
import DataProcess
import numpy as np
# %% [markdown]
# ### Task 1

# %%
#Running Task 1
print("Executing Task 1------>")
train_data, train_labels, test_data, test_labels = DataProcess.task_1_split(0.2)
#PCA=>Bayes
bayes = Bayes.Gaussian()

p_eigvecs = PCA.pca(train_data, 100)

projected_train_data = PCA.project(np.array(p_eigvecs[:,:100]).T, train_data)

bayes.train(projected_train_data, train_labels, 100)

projected_test_data = PCA.project(np.array(p_eigvecs[:,:100]).T, test_data)

print("PCA=>Bayes: {}".format(bayes.eval(projected_test_data, test_labels) * 100))

#LDA=>Bayes
_ , l_eigvecs = LDA.lda(DataProcess.list_to_dict(train_data, train_labels), 1920)

projected_train_data = PCA.project(np.array(l_eigvecs[:,:402]).T, train_data)

bayes.train(projected_train_data, train_labels, 402)

projected_test_data = PCA.project(np.array(l_eigvecs[:,:402]).T, test_data)

print("LDA=>Bayes: {}".format(bayes.eval(projected_test_data, test_labels) * 100))


#PCA=>KNN
projected_train_data = PCA.project(np.array(p_eigvecs[:,:100]).T, train_data)

projected_test_data = PCA.project(np.array(p_eigvecs[:,:100]).T, test_data)

knn = KNN.KNNClassifier(projected_train_data,train_labels,1)
print("PCA=>KNN:  {}".format(knn.eval(projected_test_data, test_labels) * 100))


#LDA=>KNN
projected_train_data = PCA.project(np.array(l_eigvecs[:,:702]).T, train_data)

projected_test_data = PCA.project(np.array(l_eigvecs[:,:702]).T, test_data)

knn = KNN.KNNClassifier(projected_train_data,train_labels,1)
print("LDA=>KNN: {}".format(knn.eval(projected_test_data, test_labels) * 100))


# %% [markdown]
# ##### Results 
# The results above show us that for task one Bayes classifier is more consistent.  Using PCA with the highest 100 eigenvalues gave the highest and most consistent results. This kept enough dimensionality that we consistently get a result in the 95%-97% range. The dimensional needs to be this high to get test results in the upper 90% otherwise it dips. Lowering it lowers the accuracy and increases the variability over different trials making it less reliable depending on the training data. I know this because the way the data is split changes each time the program runs. Something interesting is that KNN takes a long time to process. I believe it is because of the number of eigenvectors being used in the calculations. Also, interesting is that KNN classification on this data set seems to mesh well with PCA as the accuracy is 100% while it drops to 91.6% with LDA. This may not be the case always but with the given data using PCA as the reduction technique with KNN gives the best results. 
#
#

# %% [markdown]
#
# ### Task2

# %%
#Running Task2
print("Executing Task 2------>")
train_data, train_labels, test_data, test_labels = DataProcess.task_2_split(0.2)
#PCA=>Bayes


bayes = Bayes.Gaussian()

p_eigvecs = PCA.pca(train_data, 100)

projected_train_data = PCA.project(np.array(p_eigvecs[:,:100]).T, train_data)

bayes.train(projected_train_data, train_labels, 100)

projected_test_data = PCA.project(np.array(p_eigvecs[:,:100]).T, test_data)

print("PCA=>Bayes: {}".format(bayes.eval(projected_test_data, test_labels) * 100))

#LDA=>Bayes
_ , l_eigvecs = LDA.lda(DataProcess.list_to_dict(train_data, train_labels), 504)

projected_train_data = PCA.project(np.array(l_eigvecs[:,:402]).T, train_data)

bayes.train(projected_train_data, train_labels, 402)

projected_test_data = PCA.project(np.array(l_eigvecs[:,:402]).T, test_data)

print("LDA=>Bayes {}".format(bayes.eval(projected_test_data, test_labels) * 100))


#PCA=>KNN
p_eigvecs = PCA.pca(train_data, 100)

projected_train_data = PCA.project(np.array(p_eigvecs[:,:100]).T, train_data)

projected_test_data = PCA.project(np.array(p_eigvecs[:,:100]).T, test_data)

knn = KNN.KNNClassifier(projected_train_data,train_labels,14)
print("PCA=>KNN: {}".format(knn.eval(projected_test_data, test_labels) *100))

#LDA=>KNN
_ , l_eigvecs = LDA.lda(DataProcess.list_to_dict(train_data, train_labels), 504)

projected_train_data = PCA.project(np.array(l_eigvecs[:,:99]).T, train_data)

projected_test_data = PCA.project(np.array(l_eigvecs[:,:99]).T, test_data)

print("LDA=>Knn: {}".format(knn.eval(projected_test_data, test_labels) * 100))






# %% [markdown]
# ##### Results
# In task two we have results that are a lot lower than in task one. For the Bayes classifier, we see that LDA and PCA are consistently in the 80% range. This is expected as the Bayes classifier is not only accurate but also very consistent. Using KNN with this data set seems to not be very accurate at all. PCA is better than LDA but the result was subpar for KNN. I did extensive error checking and did not find any errors in my code that may be causing such low accuracy. It is possible the KNN is not a good classifier for the distribution of data in the experimental set. This shows that while KNN is good in some applications it is important to consider the type of data you are evaluating when choosing a classifier. In the task, Bayes outperformed the KNN and is more suited to this type of decision-making. 
#

# %% [markdown]
# ### Conclusion
#  In conclusion the project over all was a sucess. While it does not run as smoothly and did not give me the best results in the last task I belive over all the goal was for me to learn the aplication of classifiers to real data. We see from our test that all the classifieres are generally making good clasifications of the test data. The last test in task two using LDA for data processing and KNN as the classifier was the lowest preforming out of all the test. A 50% accuracy on classifier is considerablly random. Given more time I would take a closer look at what could be causing this issue and tr to fix or prove that it is the data not the algoritm. 
