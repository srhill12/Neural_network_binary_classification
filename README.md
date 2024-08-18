
# Neural Network Model for Binary Classification

This project demonstrates the use of a simple Keras Sequential neural network model to classify data points based on two features. The model is trained on a dataset and evaluated for its performance in predicting binary outcomes.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Overview](#data-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Analysis and Inference](#analysis-and-inference)

## Installation

To run the project, you need to have Python installed along with the required libraries. You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

## Project Structure

The project files are organized as follows:

```plaintext
├── neural_network_classification.ipynb   # Jupyter notebook with data analysis and model training
├── README.md                             # Project documentation
├── requirements.txt                      # List of required libraries
```

## Data Overview

The dataset used in this project contains 1,000 observations with two features (`Feature 1`, `Feature 2`) and a binary target variable (`Target`):

```plaintext
Feature 1  | Feature 2  | Target
-----------|------------|-------
1.2345     | -0.6789    | 0
0.1234     | 1.5678     | 1
...        | ...        | ...
```

The target variable `Target` is binary, indicating the class to which each observation belongs (0 or 1).

## Data Preprocessing

Before training the model, the data is preprocessed by scaling the features. This step ensures that the features are on a similar scale, which improves the model's convergence during training.

### Steps:

1. **Data Splitting**: The dataset is split into training and testing sets using `train_test_split`.
2. **Feature Scaling**: A `StandardScaler` is used to scale the features in both the training and testing sets.

Here’s the code snippet for preprocessing:

```python
X_scaler = StandardScaler()
X_scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

## Model Architecture

The neural network model is built using Keras' Sequential API. The model has the following architecture:

- **Input Layer**: Takes in the two features (`Feature 1` and `Feature 2`).
- **Hidden Layer**: A Dense layer with 5 units and ReLU activation function.
- **Output Layer**: A Dense layer with 1 unit and sigmoid activation function, providing a probability for the binary classification.

Here’s the code snippet for defining the model:

```python
nn_model = tf.keras.models.Sequential()
nn_model.add(tf.keras.layers.Dense(units=5, activation="relu", input_dim=input_nodes))
nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```

## Model Training

The model is compiled using the binary cross-entropy loss function and the Adam optimizer. The model is then trained for 100 epochs using the scaled training data.

### Training Code:

```python
nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
fit_model = nn_model.fit(X_train_scaled, y_train, epochs=100)
```

## Evaluation

The model's performance is evaluated on the test set, with the loss and accuracy being the primary metrics.

### Evaluation Code:

```python
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```

The final evaluation on the test set resulted in the following metrics:

```plaintext
Loss: 0.0013540424406528473
Accuracy: 1.0000
```

## Analysis and Inference

### Analysis

The neural network model achieved an accuracy of 100% on both the training and test datasets. The following factors contribute to this exceptional performance:

1. **Model Simplicity**: The model has a simple architecture with one hidden layer, which is well-suited for this relatively small and likely linearly separable dataset.
2. **Overfitting Risk**: The perfect accuracy on both the training and test sets raises concerns about potential overfitting, where the model might have memorized the training data instead of generalizing well to unseen data.
3. **Dataset Characteristics**: The scatter plot and the low loss value suggest that the classes are likely well-separated in the feature space, making it easier for the model to classify correctly.

### Inference

Given the perfect accuracy score (like a logistic regression model from a previous project), the following inferences can be drawn:

- **High Confidence**: The model's performance is indicative of a well-trained model that can classify the dataset with high confidence.
- **Potential Overfitting**: Despite the perfect accuracy, caution should be exercised as the model might not perform as well on different or more complex datasets. It is crucial to validate the model on different datasets to ensure it is not overfitting.
- **Need for Validation**: Additional techniques such as cross-validation or testing on a different validation set should be considered to ensure that the model is truly generalizing well and not simply overfitting to the given dataset.

In real-world applications, perfect accuracy is rare and often suggests that the model might not generalize well beyond the training data. Further testing and validation are recommended.
