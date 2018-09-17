import pandas as pd
import matplotlib.pyplot as plt
import random

#Read the data from the csv file
df = pd.read_csv("KNN_data.csv")
df_1 = pd.read_csv("KNN_testing.csv")

#The learning rate of the classifier
learning_rate = 0.01


#The length of the data sets
data_set_length_1 = df.shape[0]-1
data_set_length_2 = df_1.shape[0]-1


#TODO: Decide on what to do if the number of blue and red points are equal.
def nearest_neighbors(x_shift,y_shift):
	#We start the radius at 0.01 so as to not envelope a bunch of points immediately.
	r = 0.01
	blue_count = 0
	red_count = 0
	colour = 1
	
	#Picking points with increasing radius
	for i in range(100):
		
		#Checking each point 
		for j in range(data_set_length_1):
			points = df.iloc[j,:]
			x = points[0]
			y = points[1]
			label = points[2]
			value = (x - x_shift) ** 2 + (y - y_shift) ** 2 - r ** 2
			#If blue point is in the circle then do:
			if value < 0 and label == 0:
				blue_count += 1
			#If red point is in the circle then do:
			elif value < 0 and label == 1:
				red_count += 1
			if blue_count or red_count >= 2:
				if blue_count > red_count:
					colour = 0
				elif red_count < blue_count:
					colour = 1 
				break
		if blue_count or red_count >= 2:
			break
		r += learning_rate
	return(colour)

#Initialize arrays to store the testing data point info.
x_array = []
y_array = []
label_array = []

#Testing our algorithm on a data set!
for i in range(data_set_length_2):
	points = df_1.iloc[i,:]
	x = points[0]
	y = points[1]
	label = nearest_neighbors(x,y)
	
	x_array.append(x)
	y_array.append(y)
	label_array.append(label)

#Plotting the testing data points!
x = df.iloc[:,0]
y = df.iloc[:,1]
label = df.iloc[:,2]
color=['red' if l == 1 else 'blue' for l in label]
plt.scatter(x,y,color=color)

#I am defining green as red and black as blue. Using different colours 
#we can see how well our algorithm is classifying the points!

color = ['green' if l == 1 else 'black' for l in label_array]
plt.scatter(x_array,y_array,color=color)

#The point (0.8,0.2)
label = [nearest_neighbors(0.8,0.2)]
color = ['green' if l == 1 else 'black' for l in label]
plt.scatter(0.8,0.2,color=color)
#The point (0.55,0.2)
label = [nearest_neighbors(0.55,0.2)]
color = ['green' if l == 1 else 'black' for l in label]
plt.scatter(0.55,0.2,color=color)
#The point (0.2,0.4)
label = [nearest_neighbors(0.2,0.4)]
color = ['green' if l == 1 else 'black' for l in label]
plt.scatter(0.2,0.4,color=color)

plt.show()
