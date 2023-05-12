# Iris Clustering
###```The Sparks Foundation Task#2```

This code uses unsupervised machine learning to cluster the flowers in the Iris dataset into three groups, one for each species.
### Import Libraries
```python

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

```
### Read the Iris Dataset
```python
data = pd.read_csv("data/Iris.csv")

```
### Drop the Sepal Length and Width Columns
```python
data = data.drop(["SepalLengthCm", "SepalWidthCm"], axis=1)

```

### Split the Data into Training and Testing Sets
```python
xTrain, xTest, yTrain, yTest = train_test_split(data[["SepalLengthCm", "SepalWidthCm"] ],data["Species"], test_size=0.25)

```

### Create a KMeans Object with 3 Clusters
```python
kmean = KMeans(n_clusters=3)

```

### Fit the KMeans Object to the Training Data

```python
kmean.fit(xTrain)

```

### Predict the Cluster Labels for the Testing Data
```python
predection = kmean.predict(xTest)
```

### Plot the Testing Data with the Cluster Labels

```python
plt.scatter(xTest["SepalLengthCm"], xTest["SepalWidthCm"], c=predection)
plt.show()
```

The code first imports the necessary libraries. It then reads the Iris dataset into a Pandas DataFrame. The code then drops the sepal length and width columns, if desired. The code then splits the data into training and testing sets. Next, the code creates a KMeans object with 3 clusters. The object is then fit to the training data. Finally, the code predicts the cluster labels for the testing data and plots the testing data with the cluster labels.

The k-means clustering algorithm was able to successfully cluster the flowers in the Iris dataset into three groups, one for each species. This is a good example of how unsupervised machine learning can be used to find hidden patterns in data.

