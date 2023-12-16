import PIL
from PIL import Image
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

def predict_digit(image_path):
    # Load the image and convert it to grayscale
    img = Image.open(image_path).convert('L')
    # Resize the image to 8x8 pixels
    img = img.resize((8, 8), PIL.Image.LANCZOS)
    # Convert the image to a numpy array
    img = np.array(img)
    # Flatten the image to a 1D array
    img = img.reshape(64)
    # Load the digits dataset
    digits = datasets.load_digits()
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=0)
    # Create an AdaBoost classifier with a decision tree as the base estimator
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
    # Train the classifier on the training data
    clf.fit(X_train, y_train)
    # Predict the digit in the image
    digit = clf.predict([img])[0]
    return digit

# Apelați funcția predict_digit cu calea către fișierul de imagine ca argument
image_path = "C:/Users/Cosmin/Desktop/New folder/doi.png"
digit = predict_digit(image_path)
print("Cifra din imagine este:", digit)


def ex2():
    # Load the digits dataset
    digits = load_digits()

    # Reshape the data to 2D
    X = digits.data.reshape(-1, 8*8)

    # Apply k-means clustering with k=10
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)

    # Get the 10 cluster centers
    centroids = kmeans.cluster_centers_

    # Plot the centroids as images
    fig, ax = plt.subplots(2, 5, figsize=(8, 3))
    centers = centroids.reshape(10, 8, 8)
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

    plt.show()

ex2()
