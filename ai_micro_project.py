import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Loading the dataset
data = "https://raw.githubusercontent.com/gohil-jay/AI-ML-Workshop/main/dataset.csv"
dataset = pd.read_csv(data)
dataset.head()

# Visualizing the dataset
plt.hist(dataset['YearsExperience'], color='red')
plt.show()

plt.hist(dataset["Salary"], color='green')
plt.show()

var1 = dataset["YearsExperience"]
var2 = dataset["Salary"]
plt.plot(var1, var2)

# Bifurcating the dataset

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

print("X dataset --> \n")
print(X)
print("")
print("Y dataset --> \n")
print(Y)

# Splitting the bifurcated dataset(s)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

while True:
  print("Dataset split completed!")
  break

# Creating Linear Regression ML Model

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

while True:
  print("Model training completed!")
  break

# Visualizing regression line

plt.scatter(X_train, Y_train, color='green')
plt.plot(X_train, regressor.predict(X_train), color='orange')
plt.title("Salary vs Experience")
plt.xlabel("Experience (in years)")
plt.ylabel("Salary")
plt.show()

# Testing the model

Y_pred = regressor.predict(X_test) 

print("Actual values --> \n")
print(Y_test)
print("")
print("Predicted values --> \n")
print(Y_pred)

# Visualizing the test results

plt.scatter(X_test, Y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='orange')
plt.title("Salary vs Experience") 
plt.xlabel("Experience (in years)") 
plt.ylabel("Salary")
plt.show()

temp1 = []
temp2 = []
temp3 = []

for i in range(len(Y_test)):
  temp1.append(int(Y_test[i]))

for i in range(len(Y_pred)):
  temp2.append(int(Y_pred[i]))

for i in range(len(Y_pred)):
  temp3.append(i)

print("Actual values -->")
print(temp1)

print("\nPredicted values -->")
print(temp2)

print("\n")

plt.scatter(temp3, temp1, marker="*", color = 'purple')
plt.scatter(temp3, temp2, color = 'yellow')

# Evaluating the model

#Printing the prediction's confidence value

confidence = regressor.score(X_test, Y_test)
print("Confidence:         %.2f" %confidence)

# Making custom predictions

custom_pred = regressor.predict([[12]])
print("The prediction for custom input : %.5f" % custom_pred[0])

# Thank you!
