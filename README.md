# K-Nearest Neighbors (KNN) Classification on CIFAR-10 Dataset

This repository contains a project that demonstrates the application of K-Nearest Neighbors (KNN) classification on the CIFAR-10 dataset. The project involves loading the dataset, converting the images to grayscale, normalizing the images, and then using KNN for image classification.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#installation)
- [Usage](#usage)
  - [Loading CIFAR-10 Dataset](#loading-cifar-10-dataset)
  - [Data Preprocessing](#data-preprocessing)
  - [KNN Classification](#knn-classification)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images. In this project, we will perform the following steps:

1. Load the CIFAR-10 dataset.
2. Convert the color images to grayscale.
3. Normalize the grayscale images.
4. Apply K-Nearest Neighbors (KNN) classification with different distance metrics and values of K.
5. Evaluate the performance of the KNN classifier using cross-validation.

## Requirements

To run this project, you need to have Python and the following libraries installed:

- TensorFlow
- NumPy
- Matplotlib
- OpenCV
- tqdm
- scikit-learn

You can install the required libraries using pip:

```bash
pip install tensorflow numpy matplotlib opencv-python tqdm scikit-learn
```

## Usage

### Loading CIFAR-10 Dataset

The CIFAR-10 dataset can be loaded using TensorFlow's Keras API:

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
```

### Data Preprocessing

1. **Visualizing the Dataset**:

   ```python
   import matplotlib.pyplot as plt

   class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
   plt.figure(figsize=(20,8))
   for i in range(50):
       plt.subplot(5, 10, i+1)
       plt.imshow(X_train[i])
       plt.xticks([])
       plt.yticks([])
       plt.xlabel(class_names[Y_train[i][0]])
   plt.show()
   ```

2. **Converting to Grayscale**:

   ```python
   import cv2
   import numpy as np
   from tqdm import tqdm

   def convert_to_grayscale(images):
       gray_images = []
       for img in tqdm(images):
           gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
           gray_images.append(gray)
       return np.array(gray_images)

   X_train_gray = convert_to_grayscale(X_train)
   X_test_gray = convert_to_grayscale(X_test)

   # Reshape and normalize the images
   X_train_gray = X_train_gray.reshape(X_train_gray.shape[0], 32, 32, 1).astype('float32') / 255
   X_test_gray = X_test_gray.reshape(X_test_gray.shape[0], 32, 32, 1).astype('float32') / 255
   ```

### KNN Classification

1. **KNN Classifier with Cross-Validation**:

   ```python
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.model_selection import cross_val_score

   def knn_classification_cv(X_train, Y_train, metric='euclidean', k=5):
       knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)
       scores = cross_val_score(knn_classifier, X_train.reshape(len(X_train), -1), Y_train.ravel(), cv=5)
       avg_accuracy = np.mean(scores)
       return avg_accuracy

   K_values = [1, 3, 4, 5, 7, 8]
   accuracy_l1_values = []
   accuracy_l2_values = []

   for k in tqdm(K_values):
       accuracy_l1 = knn_classification_cv(X_train_gray, Y_train, metric='manhattan', k=k)
       accuracy_l1_values.append(accuracy_l1)
       accuracy_l2 = knn_classification_cv(X_train_gray, Y_train, metric='euclidean', k=k)
       accuracy_l2_values.append(accuracy_l2)
   ```

## Results

The performance of the KNN classifier with different values of K and distance metrics is evaluated using cross-validation. The results are stored in `accuracy_l1_values` and `accuracy_l2_values` for Manhattan and Euclidean distance metrics, respectively.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
