# Airline Ticket Price Prediction

## Overview
This project focuses on building and optimizing both **Machine Learning** (ML) and **Deep Learning** (DL) models to predict **airline ticket prices**. Using various regression models such as **Random Forest**, **Gradient Boosting**, **Decision Tree**, and **Lasso Regression**, as well as a deep learning model, this project aims to analyze the factors influencing ticket prices and predict the cost of tickets for new data.

## Key Features
- **Machine Learning Models**:
  - Random Forest (GridSearchCV)
  - Gradient Boosting Machine (GBM)
  - Decision Tree (GridSearchCV)
  - Lasso Regression
- **Deep Learning Model**: A neural network model for regression
- **Libraries Used**:
  - Python (TensorFlow, Keras, Scikit-learn, Pandas, Numpy)
  - Machine Learning: Random Forest, Gradient Boosting, Decision Trees, Lasso Regression
  - Deep Learning: TensorFlow, Keras
  - Data Preprocessing: Pandas, Numpy, Scikit-learn
  
- **Performance Metrics**:
  - **R²**: Coefficient of Determination (measures goodness of fit)
  - **MAPE**: Mean Absolute Percentage Error (for model accuracy)

## Models and Results

### 1. **Random Forest (GridSearchCV)**
- **R²**: 0.887 (Best performing model)
- **MAPE**: 8.05% (Indicating a high level of accuracy in predicting ticket prices)
- **Description**: Random Forest performs well on this dataset, using ensemble learning to reduce overfitting and improve prediction accuracy.

### 2. **Gradient Boosting Machine (GBM)**
- **R²**: 0.877
- **MAPE**: 12.21% (Balanced performance with early stopping)
- **Description**: Gradient Boosting offers a good balance between bias and variance, using boosting to sequentially correct errors made by previous models.

### 3. **Decision Tree (GridSearchCV)**
- **R²**: 0.814
- **MAPE**: 9.03%
- **Description**: Decision Trees perform well in terms of interpretability but show signs of overfitting, which limits their predictive ability on unseen data.

### 4. **Lasso Regression**
- **R²**: 0.424
- **MAPE**: 31.85%
- **Description**: Lasso Regression showed high bias and poor performance on this dataset, struggling to fit the model well.

### 5. **Deep Learning Model**
- **R²**: 0.577
- **MAPE**: 26.43%
- **Description**: While deep learning models are usually powerful, in this case, the structured nature of the dataset led to a less effective performance compared to traditional machine learning models.

## Requirements
To run this project, you will need the following libraries:
- Python 3.x
- TensorFlow 2.x (for deep learning models)
- Keras
- Scikit-learn
- Pandas
- Numpy

You can install all dependencies using pip:
```bash
pip install tensorflow keras scikit-learn pandas numpy
```

## Dataset
The dataset used in this project contains historical data on **airline ticket prices** and their associated features (e.g., flight date, distance, and various other factors). You can find the dataset in the repository or link to an external source if applicable.

## How to Run

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/airline-ticket-price-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd airline-ticket-price-prediction
   ```

3. Run the model training script:
   ```bash
   python train_model.py
   ```

4. After training, the model will save the trained models (Random Forest, Gradient Boosting, etc.) and their performance metrics.

5. To test predictions with new data, use:
   ```bash
   python predict.py --input new_data.csv
   ```

## Evaluation Metrics
- **R² (Coefficient of Determination)**: Measures the goodness of fit for the model. A higher value indicates better predictive performance.
- **MAPE (Mean Absolute Percentage Error)**: Measures prediction accuracy. Lower values indicate more accurate predictions.

## Acknowledgments
- The project uses **TensorFlow** and **Keras** for deep learning models and **Scikit-learn** for machine learning models and preprocessing.
- Dataset credits to the creators for providing the ticket pricing data.
