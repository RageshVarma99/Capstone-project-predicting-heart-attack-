BUSINESS OBJECTIVE/ UNDERSTANDING:

Predicting and diagnosing heart attack is the biggest challenge in the medical industry and it is based on factors like physical examination, symptoms, and signs of the patient. Factors which influence heart attack are cholesterol levels of the body, smoking habits, obesity, family history of diseases, blood pressure and working environment. Machine learning algorithms play a vital and accurate role in predicting heart attack. With the use of this project, we can use classification like ML algorithms to predict the risk of a heart attack.

Heart attack is perceived as the deadliest disease in the human life across the world. This type of disease, the heart is not capable of pushing the required quantity of blood to the remaining organs of the human body to accomplish the regular functionalities. Heart diseases are concertedly contributed by hypertension, diabetes, overweight and unhealthy lifestyle. The client wants us to predict the probability of heart attack happening to their patients. This will help them to take proactive health measures such as promoting new health related schemes.

DATA DESCRIPTION AND PREPROCESSING:

Data Dictionary:

Patient_ID: Unique ID of different patients.

Gender: Gender of the patient.

Age: Age of the patient.

HyperTension: A person has history of Hypertension or not

Heart_Disease: A person has history of heart disease or not.

Is_Married: Whether the person is married or not.

Employment_Type: Determines whether the patient is a working professional in a Private/Govt sectors, never worked or children.

Residential_type: Specifies whether the patient is from Urban/Rural areas.

Glucose_Levels: Average glucose levels of a patient.

BMI_Values: Considering height and weight of a patient.

Smoking_Habits: Classifies whether the patient is a regular smoker, past smoker or never smoked.

Heart_Attack: Chances of getting heart attack (Dependent Variable)

DATA PREPARATION:

Null values in the column are replaced by median value and stored.
Null values in ‘Smoking_Habits’ are filled with ‘never smoked’.
Null value imputation done for both the independent variables.

Outlier treatment:

We plot boxplot on log transformed values of BMI_Values. Outlier values are also displayed.
Eliminating outliers based on IQR technique. When we plot boxplot, the outliers are neglected.
We plot distplot on outlier treated BMI_Values and we get a near normal distribution.
Skewness of Glucose_Levels is very high. We plot boxplot on the column. We infer that this column has many outliers too.
Glucose_Levels is highly positively skewed. By using skew function.

EXPLORATORY DATA ANALYSIS & BUSINESS INSIGHTS:

Multi-collinearity:

As VIF values are less than 5, we can conclude that there is no multi collinearity amongst independent variables.

BASIC MODEL:

Accuracy score for Logistic Regression method is 0.8578.

Accuracy score for Decision Tree method is 0.9716.

Accuracy score for Random Forest method is 0.9754.

Transformation:
All the categorical variables are stored in a new dataframe named df1_cat.
Dummy variables are created for all categorical variables using get_dummies function from pandas and all encoded columns are stored in df1_cat. .(one hot encoding).
Scaling the data:
Performing standardization on numerical dataframe num_df using standard scaler function from sklearn.preprocessing module.
We are concatenating the scaled numerical variables and encoded categorical variables and storing it in df_pre.df_pre is the preprocessed dataframe ready for modelling.
Concatenating preprocessed df_new with target variable to form a dataframe df_corr.We are plotting heatmap on df_corr to find correlation amongst two independent variables.

HYPER PARAMETERS TUNING:

We have found out the best parameters using Grid Search CV. The parameters are as follows:
'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': 9, 'min_samples_leaf': 1,
'minsamples_split': 2, 'n_estimators': 30

Age is the most important feature in prediction of Heart attack of a patient.

COMPARISON AND SELECTION OF MODEL:

Based on our comparison done on all the Classification related algorithms, we found out that Random Forest model gives us the best accuracy score of 0.9754 and has the least False Negative values. Random Forest has the highest accuracy in predicting the correct classes.

RESULTS & DISCUSSION:

We proposed three methods in which comparative analysis was done and promising results were achieved. The conclusion which we found is that Random Forest machine learning algorithm performed better in this analysis.

DESCRIPTION OF CRITERION:

The methods which are used for comparison are Confusion Matrix, Precision, Specificity, Sensitivity, and F1 score. For some features which were in the dataset, Random Forest and decision tree classifier algorithms performed better in the ML approach when data preprocessing is applied.
The dataset size can be increased and then Machine learning with various other optimizations can be used and more promising results can be achieved.
Machine learning and various other optimization techniques can also be used so that the evaluation results can again be increased. More different ways of normalizing the data can be used and the results can be compared. And more ways could be found where we could integrate heart-disease-trained ML models with certain multimedia for the ease of patients.
