# Heart Disease Prediction

## Project Overview

This project focuses on predicting whether a person has had a heart attack using health and lifestyle information.

The project uses the **Personal Key Indicators of Heart Disease** dataset from Kaggle. Different machine learning techniques are applied to explore the data, preprocess it, handle class imbalance, train classification models, and evaluate their performance.

The main machine learning models used in this project are:

* Decision Tree
* Random Forest
* Random Forest with SMOTE

The final model can also be tested with new user-provided health information to generate a prediction.

---

## Dataset

The dataset is obtained from Kaggle:

**Personal Key Indicators of Heart Disease**

Dataset source:

* Kaggle: `kamilpytlak/personal-key-indicators-of-heart-disease`
* Dataset file: `heart_2022_no_nans.csv`

The dataset contains health, lifestyle, demographic, and medical information about individuals.

The target variable is:

```text
HadHeartAttack
```

This variable indicates whether the person has had a heart attack.

---

## Project Workflow

The project follows these main steps:

```text
Dataset
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature & Target Separation
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Class Imbalance Handling
   ↓
Model Training
   ↓
Model Prediction
   ↓
Model Evaluation
   ↓
Test with New User Data
```

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn

### Environment

* Google Colab
* Kaggle Dataset

---

# 1. Dataset Download

The Kaggle API is used to download the dataset directly into Google Colab.

```python
!pip install -q kaggle

!kaggle datasets download -d kamilpytlak/personal-key-indicators-of-heart-disease
```

The downloaded ZIP file is then extracted:

```python
import zipfile

with zipfile.ZipFile("personal-key-indicators-of-heart-disease.zip", "r") as zip_ref:
    zip_ref.extractall("heart-disease")
```

The dataset is loaded using Pandas:

```python
df = pd.read_csv("heart-disease/2022/heart_2022_no_nans.csv")
```

---

# 2. Data Cleaning

The first step is to understand the dataset and check its structure.

### Dataset Preview

```python
df.head()
```

### Dataset Shape

```python
df.shape
```

### Missing Values

```python
df.isnull().sum()
```

The dataset used in this project does not contain missing values.

### Duplicate Records

Duplicate records are checked using:

```python
df.duplicated().sum()
```

Duplicate rows are removed:

```python
df.drop_duplicates(inplace=True)
```

### Dataset Information

```python
df.info()
```

This helps identify:

* Number of records
* Number of features
* Data types
* Memory usage

---

# 3. Exploratory Data Analysis

Exploratory Data Analysis (EDA) is performed to understand the dataset and identify patterns in the variables.

## Numerical Feature Distribution

Histograms are used to understand the distribution of numerical features.

```python
numerical_cols = df.select_dtypes(
    include=['int64', 'float64']
).columns

df[numerical_cols].hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()
```

Histograms help identify:

* Data distribution
* Skewness
* Possible unusual values
* General range of numerical variables

---

## Correlation Analysis

A correlation heatmap is used to understand relationships between numerical variables.

```python
plt.figure(figsize=(8, 6))

sns.heatmap(
    df[numerical_cols].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()
```

Correlation analysis helps identify variables that have stronger or weaker relationships with each other.

---

## Outlier Detection

Boxplots are used to identify possible outliers in numerical features.

```python
for col in numerical_cols:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()
```

Boxplots provide information about:

* Median
* Quartiles
* Data spread
* Potential outliers

---

## Categorical Features vs Target

Categorical variables are also compared with the target variable.

For example, the relationship between sex and heart attack history is visualized using:

```python
plt.figure(figsize=(6, 4))

sns.countplot(
    x='Sex',
    hue='HadHeartAttack',
    data=df
)

plt.show()
```

This helps understand how different categories are distributed across the target classes.

---

# 4. Feature and Target Separation

The target variable is separated from the input features.

```python
x = df.drop("HadHeartAttack", axis=1)
y = df["HadHeartAttack"]
```

Here:

* `x` contains the input features.
* `y` contains the target variable.

---

# 5. Data Encoding

The dataset contains categorical variables, so they need to be converted into numerical values before training machine learning models.

## Target Encoding

The target variable is encoded using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)
```

## Feature Encoding

Categorical columns are also converted into numerical values.

```python
for col in x.select_dtypes(include="object").columns:
    x[col] = LabelEncoder().fit_transform(x[col])
```

This converts categorical values into numerical representations that machine learning models can process.

---

# 6. Train-Test Split

The dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

The `stratify=y` parameter helps maintain a similar class distribution in both training and testing datasets.

---

# 7. Machine Learning Models

Two main classification models are trained initially.

## Decision Tree

A Decision Tree is trained with class balancing:

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

dt.fit(x_train, y_train)
```

The `class_weight="balanced"` parameter gives more importance to the minority class.

---

## Random Forest

A Random Forest model is also trained:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

rf.fit(x_train, y_train)
```

Random Forest combines multiple decision trees to make predictions.

---

# 8. Model Prediction

Predictions are generated for the test dataset.

### Decision Tree

```python
y_pred = dt.predict(x_test)
```

### Random Forest

```python
rf_pred = rf.predict(x_test)
```

---

# 9. Model Evaluation

The models are evaluated using several classification metrics.

The metrics used are:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

## Accuracy

Accuracy represents the percentage of total predictions that are correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

---

## Precision

Precision measures how many of the samples predicted as positive are actually positive.

```text
Precision = TP / (TP + FP)
```

---

## Recall

Recall measures how many actual positive cases were correctly identified.

```text
Recall = TP / (TP + FN)
```

For medical prediction problems, recall is particularly important because missing a positive case can be significant.

---

## F1 Score

F1 Score combines precision and recall into a single metric.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## ROC-AUC

ROC-AUC measures how well the model separates the two classes across different classification thresholds.

The probability predictions are obtained using:

```python
y_prob = rf.predict_proba(x_test)[:, 1]
```

Then ROC-AUC is calculated:

```python
roc_auc_score(y_test, y_prob)
```

---

# 10. Handling Class Imbalance

The target classes are not equally distributed, which can affect model performance.

First, the class distribution is checked:

```python
pd.Series(y_train).value_counts()
```

Two approaches are used in the project:

1. Class weighting
2. SMOTE

---

# 11. SMOTE

SMOTE stands for **Synthetic Minority Over-sampling Technique**.

It creates synthetic examples of the minority class instead of simply duplicating existing samples.

SMOTE is applied only to the training data:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

x_train_smote, y_train_smote = smote.fit_resample(
    x_train,
    y_train
)
```

The class distribution is then checked before and after SMOTE:

```python
print("Before SMOTE:")
print(pd.Series(y_train).value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())
```

This helps create a more balanced training dataset.

---

# 12. Random Forest with SMOTE

A new Random Forest model is trained using the SMOTE-balanced training data.

```python
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(x_train_smote, y_train_smote)
```

The model is then used to make predictions:

```python
y_pred = rf.predict(x_test)
```

Probability scores are also generated:

```python
y_prob = rf.predict_proba(x_test)[:, 1]
```

---

# 13. Final Model Evaluation

The Random Forest + SMOTE model is evaluated using:

```python
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
```

A classification report is also generated:

```python
print(classification_report(y_test, y_pred))
```

The confusion matrix is calculated using:

```python
confusion_matrix(y_test, y_pred)
```

---

# 14. Confusion Matrix

A confusion matrix shows how many predictions fall into each category.

It contains:

* True Positive (TP)
* True Negative (TN)
* False Positive (FP)
* False Negative (FN)

The confusion matrix helps understand which types of predictions the model gets right or wrong.

---

# 15. Testing the Model with New Data

The trained model can also be tested using new health information.

Example:

```python
user_input = pd.DataFrame({
    "State": [17],
    "Sex": [1],
    "GeneralHealth": [4],
    "PhysicalHealthDays": [0.0],
    "MentalHealthDays": [4.0],
    "LastCheckupTime": [3],
    "PhysicalActivities": [1],
    "SleepHours": [6.0],
    "RemovedTeeth": [2],
    "HadAngina": [0],
    "HadStroke": [0],
    "HadAsthma": [0],
    "HadSkinCancer": [0],
    "HadCOPD": [0],
    "HadDepressiveDisorder": [0],
    "HadKidneyDisease": [0],
    "HadArthritis": [0],
    "HadDiabetes": [0],
    "DeafOrHardOfHearing": [0],
    "BlindOrVisionDifficulty": [0],
    "DifficultyConcentrating": [0],
    "DifficultyWalking": [0],
    "DifficultyDressingBathing": [0],
    "DifficultyErrands": [0],
    "SmokerStatus": [0],
    "ECigaretteUsage": [0],
    "ChestScan": [0],
    "RaceEthnicityCategory": [0],
    "AgeCategory": [0],
    "HeightInMeters": [1.70],
    "WeightInKilograms": [58.97],
    "BMI": [20.36],
    "AlcoholDrinkers": [1],
    "HIVTesting": [0],
    "FluVaxLast12": [1],
    "PneumoVaxEver": [0],
    "TetanusLast10Tdap": [2],
    "HighRiskLastYear": [0],
    "CovidPos": [0]
})
```

The model prediction is generated using:

```python
prediction = rf.predict(user_input)[0]
```

The class probabilities are also displayed:

```python
probability = rf.predict_proba(user_input)[0]

print("Prediction:", prediction)
print("Probability of Class 0:", probability[0])
print("Probability of Class 1:", probability[1])
```

This allows the model to provide both the predicted class and the estimated probability for each class.

---

# 16. Comparing Predicted and Actual Values

The model can also be tested against an existing sample from the test dataset.

```python
sample = x_test.iloc[[0]]

prediction = rf.predict(sample)[0]

print("Predicted:", prediction)
print("Actual   :", y_test[0])
```

This provides a simple way to compare the model's prediction with the actual target value.

---

# 17. Project Structure

A possible project structure is:

```text
Heart-Disease-Prediction/
│
├── heart_disease_prediction.ipynb
├── README.md
├── requirements.txt
│
└── dataset/
    └── heart_2022_no_nans.csv
```

If the dataset is downloaded directly from Kaggle, it does not need to be included in the GitHub repository.

---

# 18. Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd Heart-Disease-Prediction
```

Install the required libraries:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

For Google Colab, the required packages can be installed directly in the notebook.

---

# 19. How to Run

### Using Google Colab

1. Open the notebook in Google Colab.
2. Upload your Kaggle API JSON file.
3. Run the Kaggle dataset download section.
4. Extract the dataset.
5. Run the data loading and cleaning sections.
6. Perform EDA.
7. Preprocess the data.
8. Train the models.
9. Evaluate the models.
10. Test the model with sample user data.

---

# 20. Key Machine Learning Concepts Used

This project demonstrates several important data science and machine learning concepts:

* Data collection
* Data cleaning
* Exploratory Data Analysis
* Numerical feature analysis
* Categorical feature analysis
* Correlation analysis
* Outlier detection
* Label encoding
* Train-test splitting
* Class imbalance
* Class weighting
* SMOTE
* Decision Tree
* Random Forest
* Model prediction
* Classification metrics
* Confusion Matrix
* ROC-AUC
* Probability prediction

---

# 21. Limitations

This project is developed for educational and machine learning practice purposes.

The model should **not be used as a medical diagnosis system**. Predictions are based on patterns learned from the dataset and do not replace professional medical evaluation.

The dataset, preprocessing method, feature representation, and model performance can also affect the reliability of predictions.

---

# 22. Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* Feature selection
* Feature engineering
* Comparing additional machine learning models
* XGBoost implementation
* Logistic Regression comparison
* ROC curve visualization
* Precision-Recall curve
* Feature importance analysis
* SHAP explainability
* Model deployment using FastAPI or Flask
* Creating a web interface for prediction
* Saving and loading the trained model
* Building a complete prediction API

---

# 23. Conclusion

This project demonstrates an end-to-end machine learning workflow for heart disease prediction.

The process starts with downloading and understanding the dataset, followed by data cleaning and exploratory data analysis. The categorical variables are encoded, the dataset is divided into training and testing sets, and different classification models are trained.

Class imbalance is addressed using class weighting and SMOTE. Decision Tree and Random Forest models are evaluated using accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix.

The project also demonstrates how a trained Random Forest model can be used to make predictions on new health-related input data.

---

## Disclaimer

This project is for **educational and research purposes only**. It is not intended to provide medical diagnosis, treatment, or professional medical advice.
