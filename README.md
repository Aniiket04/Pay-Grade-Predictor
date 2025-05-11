# Pay-Grade-Predictor Project

## Project Overview 

**Project Title : Pay-Grade-Predictor**:
The goal of the project is to develop a machine learning model that predicts whether an individual's annual salary exceeds $100,000 based on factors such as:
Company: The organization the person works for.
Job Role: The specific position or job title the individual holds.
Education Level: Whether the individual has a bachelor's degree or a master's degree

## Objectives
1. **Predict Salary Threshold**:
Build a machine learning model to classify individuals into two categories:
Annual salary > $100,000.
Annual salary ≤ $100,000.

2. **Feature Analysis**:
Analyze how features such as company, job role, and education level influence the likelihood of exceeding the $100,000 salary threshold.

## Project Structure

### 1. Importing Libraries
The notebook begins by importing essential Python libraries, including:
pandas for data manipulation
numpy for numerical operations
sklearn (from scikit-learn) for machine learning tools
```python
import pandas as pd
import numpy as np
from sklearn import linear_model
```

### 2. Loading the Dataset
The given dataset is loaded using pandas.read_csv(). This dataset contains data about company, job,	degree and whether salary is more than 100k or not
```python
df=pd.read_csv("Data4.csv")
df.columns
```

### 3. Data processing
3.1 Process the given dataset by separating the features and target variables for the machine learning task.
```python
input=df.drop(["salary_more_than_100k"],axis='columns')
target=df["salary_more_than_100k"]
input
```
3.2 Set up label encoders to convert categorical variables into numeric values
```python
from sklearn.preprocessing import LabelEncoder
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
input['company_n']=le_company.fit_transform(input['company'])
input['job_n']=le_job.fit_transform(input['job'])
input['degree_n']=le_degree.fit_transform(input['degree'])
df
input
```

### 4.Model Training
From scikit learn we import the tree module for building decision tree models
```python
from sklearn import tree
input_n=input.drop(['company','job','degree'],axis='columns')
input_n
model=tree.DecisionTreeClassifier()
model.fit(input_n,target)
```

### 5. Model Predictions
Predictions on the held-out dataset.
```python
model.predict([[0,2,0]])
model.predict([[2,1,1]])
model.score(input_n,target)
```
The model.score() function evaluates the performance of a trained model.

## Conclusion
This project successfully implemented a machine learning model to predict whether an individual's annual salary exceeds $100,000 based on their company, job role, and education level.

## Author - Aniket Pal
This project is part of my portfolio, showcasing the machine learning skills essential for data science roles.

-**LinkedIn**: [ www.linkedin.com/in/aniket-pal-098690204 ]
-**Email**: [ aniketspal04@gmail.com ]



