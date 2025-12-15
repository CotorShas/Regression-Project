#LRcode.py          Schuyler Hayes              10-2025
#Liinear Regression prediction for vehicles prices for CPS 493 class

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

carData = pd.read_csv("vehicle_price_prediction.csv", keep_default_na=False)

#print(carData.head())  #just testing to see if it was importated correctly

#carData.columns
randomtestnumber = 44 #I have set the random state number to bee easily changeable, that way I Can easily compre differnt random selections of the data

carDataSample = carData.sample(n=10000, random_state=randomtestnumber)  #cutting 10k randome samples, with a set state so I can use the same random samples in each regression model (I chose to use 44 because its my house number)

#print(carDataSample.head()) another test


#print(sampleDataRandomized.head())

#finding and removing outliers
average = carData['price'].mean()
standardDeviation = carData['price'].std()

upper = average + 2 * standardDeviation
lower = average - 2 * standardDeviation

cleanedSample = carDataSample[
    (carDataSample['price'] >= lower) &
    (carDataSample['price'] <= upper)
]

sampleDataRandomized = cleanedSample.sample(frac=1, random_state=randomtestnumber).reset_index(drop=True)  #randomizing the selected samples, restting index to ordered list from 0 to 9999

x = sampleDataRandomized.iloc[:,0:19] #features, our x variable(s)
y = sampleDataRandomized.iloc[:,19]   #price, our y variable


#x = sampleDataRandomized.drop(columns=['column'])  #allowing us to drop partiuclar columns for testing purposes
#x = sampleDataRandomized.drop(columns=['make','model', 'exterior_color', 'interior_color'])  #dropping nominal columns to see how it effects the linear regression

#print(x.head())
#print(y.head())



#print(x['accident_history'].unique())

#replacing natually ordered catagorical data with nummerical values
#x['condition'] = x['condition'].map({'Fair': 1, 'Good': 2, 'Excellent': 3})
#x['accident_history'] = x['accident_history'].map({'None': 3, 'Minor': 2, 'Major': 1})
#x['trim'] = x['trim'].map({'Base': 1, 'LX': 2, 'Sport': 3, 'EX': 4, 'Touring': 5, 'Limited': 6})


#removing nominal data for testing
#x.drop(columns=['make', 'model', 'transmission', 'fuel_type', 'exterior_color', 'interior_color', 'seller_type'], inplace=True)


#we now need to convert catagorical data to nummerical (e.g car brand to generric number)
xNumbers = pd.get_dummies(x, drop_first=True)  




XTrain, XTestStorage, YTrain, YTestStorage = train_test_split(xNumbers, y, test_size=0.3, random_state=randomtestnumber)  #splitting data into 70/30 training testing
XVal, XTest, YVal, YTest = train_test_split(XTestStorage, YTestStorage, test_size=0.5, random_state=randomtestnumber) #splitting trainingstorage into 50/50 (thus 15/15) test validation

#create anf fit the model
model = LinearRegression()  
model.fit(XTrain, YTrain)  


#validation test
predictions = model.predict(XVal)  

#test using MSE
mse = mean_squared_error(YVal, predictions)
r2 = r2_score(YVal, predictions)
print("validation")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")



#final teest
testPredictions = model.predict(XTest)  

#test using MSE
mseTest = mean_squared_error(YTest, testPredictions)
r2Test = r2_score(YTest, testPredictions)
print("test")
print(f"Mean Squared Error: {mseTest}")
print(f"R^2 Score: {r2Test}")
