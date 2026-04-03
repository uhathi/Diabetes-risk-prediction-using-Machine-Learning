## Diabetes Risk Prediction using Machine Learning ##

## Overview
This project aims to predict the risk of diabetes in individuals using machine learning classification models. The focus is on accurately identifying high-risk patients (minority class) while maintaining overall model performance.


## Objective
* Build multiple classification models to predict diabetes risk
* Address class imbalance in the dataset
* Compare model performance using appropriate evaluation metrics
* Select the best model for identifying high-risk individuals


## Models Implemented
The following machine learning models were developed and evaluated:
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Support Vector Machine (SVM)
* Ensemble Model (Soft Voting Classifier)


## Methodology

### 1. Data Preprocessing
* Train-test split using **stratified sampling** to preserve class distribution
* Feature scaling using **StandardScaler** for distance-based models


### 2. Baseline Modeling
* Initial models were built using default parameters
* Provided a benchmark for evaluating improvements


### 3. Hyperparameter Tuning
* Used **GridSearchCV** to optimize model performance
* Evaluation metric: **F1-score** (due to class imbalance)


### 4. Final Model Training
* Best hyperparameters were selected from GridSearchCV
* Models retrained on training data


### 5. Ensemble Model
* Built a **heterogeneous ensemble** combining:
  * Logistic Regression
  * KNN
  * SVM
  * Decision Tree
* Used **soft voting** to aggregate predicted probabilities


## Evaluation Metrics
Models were evaluated using:
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
Since the dataset is imbalanced, special emphasis was placed on **Recall and F1-score for the diabetic (minority) class**


## Results Summary
* **Support Vector Machine (SVM)** achieved the best overall performance with a good balance between recall for diabetic class, F1-score and ROC-AUC.

* **ensemble model** achieved higher overall accuracy but lower recall for diabetic cases


## Final Model Selection
**SVM was selected as the final model** as it provided the best trade-off between detecting diabetic individuals and maintaining overall predictive performance.


## Key Insights
* KNN performed poorly due to sensitivity to class imbalance
* Decision Tree achieved high recall but lower precision (more false positives)
* Ensemble model did not outperform SVM due to inclusion of weaker models


## Technologies Used
* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib


## Future Improvements
* Apply resampling techniques (SMOTE, undersampling)
* Experiment with weighted ensemble models
* Explore advanced models like XGBoost or LightGBM


