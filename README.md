# Project Overview
The objective of this project is to build a deep learning model that can recognize handwritten digits. We utilize the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits (0-9). This project uses TensorFlow and Keras to design, train, and evaluate a neural network model for digit classification.
# Technologies Used
Python
TensorFlow
Keras
Matplotlib (for visualizing results)
NumPy (for data manipulation)
# Dataset
We used the popular MNIST dataset for handwritten digits, which is available directly from tensorflow.keras.datasets. The images are grayscale and of size 28x28 pixels. The dataset is divided into:
Training set: 60,000 images
Test set: 10,000 images
# Model Architecture
The neural network has the following architecture:
Flatten Layer: Converts the 28x28 pixel input images into a flat vector of size 784.
Dense Layer: A fully connected layer with 128 neurons and ReLU activation.
Dropout Layer: Dropout rate of 0.2 to prevent overfitting.
Output Layer: A softmax output layer with 10 neurons for digit classification (0-9).
# Model Compilation:
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metrics: Accuracy
# Results
After training, the model achieves an accuracy of approximately 99% on the training set and 98% on the test set. Below are some examples of the model's predictions on test images:
