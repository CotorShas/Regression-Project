#RRcode.py          Schuyler Hayes              10-2025
#Ridge Regression prediction for vehicles prices for CPS 493 class

import pandas as pd
import numpy as np
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

carData = pd.read_csv("vehicle_price_prediction.csv", keep_default_na=False)

#print(carData.head())  #just testing to see if it was importated correctly

#carData.columns
randomtestnumber = 124 #I have set the random state number to bee easily changeable, that way I Can easily compre differnt random selections of the data


carDataSample = carData.sample(n=10000, random_state=randomtestnumber)  #cutting 10k randome samples, with a set state so I can use the same random samples in each regression model (I chose to use 44 because its my house number)

#print(carDataSample.head()) another test

sampleDataRandomized = carDataSample.sample(frac=1, random_state=randomtestnumber).reset_index(drop=True)  #randomizing the selected samples, restting index to ordered list from 0 to 9999

#print(sampleDataRandomized.head())


x = sampleDataRandomized.iloc[:,0:19] #features, our x variable(s)
y = sampleDataRandomized.iloc[:,19]   #price, our y variable

#print(x.head())
#print(y.head())


#replacing natually ordered catagorical data with nummerical values
x['condition'] = x['condition'].map({'Fair': 1, 'Good': 2, 'Excellent': 3})
x['accident_history'] = x['accident_history'].map({'None': 1, 'Minor': 2, 'Major': 3})
x['trim'] = x['trim'].map({'Base': 1, 'LX': 2, 'Sport': 3, 'EX': 4, 'Touring': 5, 'Limited': 6})

#we now need to convert catagorical data to nummerical (e.g car brand to generric number)
xNumbers = pd.get_dummies(x, drop_first=True)  

XTrain, XTest, YTrain, YTest = train_test_split(xNumbers, y, test_size=0.2, random_state=randomtestnumber)  #splitting data into 80/20 training testing


model = linear_model.Ridge(alpha=0.1) #choosing/making the model
model.fit(XTrain, YTrain)

predictions = model.predict(XTest)

mse = mean_squared_error(YTest, predictions)
r2 = r2_score(YTest, predictions)


print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


