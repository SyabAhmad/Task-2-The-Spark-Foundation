import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/Iris.csv")

print(data.columns)

# data = data.drop(["SepalLengthCm", "SepalWidthCm"], axis=1)

xTrain, xTest, yTrain, yTest = train_test_split(data[["SepalLengthCm", "SepalWidthCm"] ],data["Species"], test_size=0.25)

kmean = KMeans(n_clusters=3)

kmean.fit(xTrain)

predection = kmean.predict(xTest)

plt.scatter(xTest["SepalLengthCm"], xTest["SepalWidthCm"], c=predection)

plt.show()

print(predection)
