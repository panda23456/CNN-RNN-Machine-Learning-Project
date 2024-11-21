# CA1: Convolutional and Recurrent Neural Networks

## Project Overview
This project involves implementing and evaluating deep learning models, including Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), for tasks such as image classification and sentiment analysis. The focus is on data preprocessing, feature engineering, model building, and performance improvement through augmentation and hyperparameter tuning.

## Objectives
### Part A: Convolutional Neural Network (CNN)
1. **Background Research**:
   - Organize datasets (Train, Validation, and Test folders) and ensure consistency in image labels and distribution.
   - Address data quality issues (e.g., mislabeled images in folders).

2. **Data Preprocessing**:
   - Convert images to grayscale and resize to standard dimensions (37x37 and 131x131).
   - Visualize data to identify potential issues affecting model accuracy.

3. **Feature Engineering**:
   - Augment data to balance class distributions, focusing on underrepresented categories like radishes.
   - Apply data augmentation to all vegetable classes for improved generalization.

4. **Model Evaluation**:
   - Train a base CNN model and analyze the confusion matrix to identify misclassifications.
   - Augment data and observe changes in performance, with improvements noted for specific classes.

5. **Model Improvement**:
   - Use hyperparameter tuning (GridSearch) to optimize model configurations.
   - Apply callbacks such as learning rate adjustment and early stopping for better training efficiency.

### Part B: Recurrent Neural Network (RNN)
1. **Exploratory Data Analysis**:
   - Analyze text datasets for sentiment prediction tasks in multiple languages (English, Malay, and Chinese).

2. **Feature Engineering**:
   - Visualize word importance to understand patterns in text data.

3. **Model Training and Evaluation**:
   - Train a base RNN model for sentiment classification.
   - Analyze confusion matrices to evaluate performance across different datasets.

4. **Model Improvement**:
   - Perform hyperparameter tuning using GridSearch.
   - Enhance model performance through callbacks for learning rate adjustment and early stopping.

## Key Findings
### Part A: CNN
- Data augmentation significantly improved class-specific predictions (e.g., radish and brinjal accuracy).
- Misclassifications often occurred between visually similar vegetables (e.g., cauliflower and cabbage).
- Hyperparameter tuning and augmenting all vegetable classes improved overall model accuracy.

### Part B: RNN
- Initial models struggled with multilingual sentiment prediction, showing a bias toward negative reviews.
- Adding more training data and tuning hyperparameters reduced prediction errors and improved metrics.
- Callback mechanisms enhanced model efficiency and minimized overfitting.

## Tools and Technologies
- Python libraries: TensorFlow, Keras, NumPy, Matplotlib
- Data preprocessing: Image resizing, grayscale conversion, and augmentation
- Model evaluation: Confusion matrices and performance metrics

## Summary
This project demonstrates the importance of:
- Addressing data quality issues and using preprocessing techniques for better model performance.
- Leveraging data augmentation and hyperparameter tuning to optimize deep learning models.
- Using callbacks for efficient and robust training processes.

## Author
Loh Yip Khai

