  
"""
-------------------------------------------------------------------
-- Title:
-- File:    main.py
-- Purpose: Analysis of the dataset & development of the machine learning
            algorithms to predict which clients are most likely to subsribe
            to a product
-- Author:  Georgios Spyrou
-- Date:    08/08/2020
-------------------------------------------------------------------
"""

# Import dependencies
import os
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sns.set_style("dark")


# Location of the project folder and the dataset (csv file)
project_folder = r'C:\Users\george\Desktop\GitHub\Projects\Marketing_Campaigns'

os.chdir(project_folder)

data_file_loc = os.path.join(project_folder, 'Data',
                             'bank_marketing_campaigns_full.csv')

bank_marketing_df = pd.read_csv(data_file_loc, sep=';', header=[0])


# Part-1: Data Cleaning and Exploratory Data Analysis

bank_marketing_df.columns

# The last column indicates if the client subscribed a term deposit or not
# But the current name is 'y' which is not very explanatory - so we rename it
bank_marketing_df.rename(columns={'y': 'subscribed'}, inplace=True)

# Data Cleaning

# Check if the dataframe has duplicates and remove them
df_shape = bank_marketing_df.shape
print(f'There are {df_shape[0]} rows and {df_shape[1]} columns in the dataset')

bank_marketing_df = bank_marketing_df.drop_duplicates()
df_shape = bank_marketing_df.shape
print(f'''There are {df_shape[0]} rows and {df_shape[1]} columns in the
                     dataset after removing duplicates''')

# Identify how many missing values we have per column
bank_marketing_df.isnull().sum(axis=0)

# There are many missing values in other forms like 'unknown'
unknown_cols = bank_marketing_df.isin(['unknown']).sum(axis=0)
unknown_cols[unknown_cols > 0]

# Exploratory Data Analysis

# We import a custom package made for this project and contains all the
# functions used
from Functions import marketing_campaigns_functions as mcf

# Visualize the distribution of the number of clients who subscribed
mcf.visualizeCategoricalVar(bank_marketing_df, col='subscribed', hue=None,
                            figsize=(10, 6))

# We can observe that the dataset is hignly imbalanced towards the clients
# who did not subscribe for the product


# Categorical Variables

# Visualize the Categorical variables
mcf.visualizeCategoricalVar(bank_marketing_df, figsize=(14, 12), col='job',
                            hue='subscribed')

mcf.visualizeCategoricalVar(bank_marketing_df, figsize=(14, 12), col='marital',
                            hue='subscribed')

mcf.visualizeCategoricalVar(bank_marketing_df, figsize=(14, 12),
                            col='education', hue='subscribed')

mcf.visualizeCategoricalVar(bank_marketing_df, figsize=(14, 12), col='contact',
                            hue='subscribed')

mcf.visualizeCategoricalVar(bank_marketing_df, figsize=(14, 12),
                            col='poutcome', hue='subscribed')

mcf.visualizeCategoricalVar(bank_marketing_df, figsize=(14, 12), col='month',
                            hue='subscribed')

mcf.visualizeCategoricalVar(bank_marketing_df, figsize=(14, 12),
                            col='day_of_week', hue='subscribed')


# Numerical Variables

# Have a look at some statistics for the numerical variables
summary_num_vars = bank_marketing_df.describe()

# Age
mcf.visualizeNumericalVar(input_df=bank_marketing_df, col='age',
                          hue='subscribed', figsize=(12, 10))

# Duration
mcf.visualizeNumericalVar(input_df=bank_marketing_df, col='duration',
                          hue='subscribed', figsize=(12, 10))

# Campaign
mcf.visualizeNumericalVar(input_df=bank_marketing_df, col='campaign',
                          hue='subscribed', figsize=(12, 10))

# pdays
bank_marketing_df.groupby('pdays').count()


# Generate a pair plot and a heatmap to identify the correlation
# Correlation is a statistical measure that indicates the extend to which two
# or more variables fluctuate together.

summary_num_vars = bank_marketing_df.describe()

# Pairplot
def hide_upper_triangle(*args, **kwds):
    plt.gca().set_visible(False)


plt.figure(figsize=(18, 14))
pair_plt = sns.pairplot(data=bank_marketing_df, vars=summary_num_vars.columns,
                        hue='subscribed', diag_kind='hist',
                        palette={"no": "darkcyan", "yes": "firebrick"})
pair_plt.map_upper(hide_upper_triangle)
pair_plt.add_legend()
pair_plt.set(alpha=0.5)
pair_plt.fig.suptitle('Pair plot for Numerical Variables', fontweight="bold")
plt.show()

# Heatmap

# Calculate the Pearson Correlation Coefficients
corr = bank_marketing_df[summary_num_vars.columns].corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, cmap='Reds', vmax=1.0, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Its clear the most of the values in the pdays column is 999 indicating that
# the client was not previously contacted
# Therefore we will change this variable and split it in 3 buckets
bank_marketing_df['pdays_bucket'] = bank_marketing_df['pdays'].apply(
        lambda x: mcf.bucketPday(x))

bank_marketing_df.groupby('pdays_bucket').count()
bank_marketing_df.drop(columns=['pdays'], inplace=True)


# Transform the 'yes' and 'no' values (target variable) to 1 and 0 respectively
bank_marketing_df['subscribed'] = bank_marketing_df['subscribed'].map(
        {'yes': 1, 'no': 0})

# Part-2: Data Preprocessing

# We have a few categorical variables that need encoding before fit them in
# the model --> one hot encoding
categorical_cols = bank_marketing_df.select_dtypes(include=['object']).columns

encoded_df = pd.concat([bank_marketing_df, pd.get_dummies(
        bank_marketing_df[categorical_cols])], axis=1)

encoded_df = encoded_df.drop(categorical_cols, axis=1)
encoded_df.info()

# Data normalization for the numerical features
# All values will fall in the range between 0 and 1

# Scale
scaler = MinMaxScaler()
encoded_scaled_df = pd.DataFrame(scaler.fit_transform(encoded_df),
                                 columns=encoded_df.columns)

# Split the data to train and test sets
X = encoded_scaled_df.loc[:, encoded_scaled_df.columns != 'subscribed']
y = encoded_scaled_df['subscribed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

y_train.value_counts()
y_test.value_counts()


# Part-3: Model creation, tuning and evaluation

# Model A: Random Forest with all 20 predictor variables

# Tune the Random Forest Model
param_grid = {
    'bootstrap': [True],
    'max_features': [10, 16, 18, 20],
    'min_samples_leaf': [3, 6, 9, 12],
    'min_samples_split': [5, 10, 15, 20],
    'n_estimators':  [100, 200, 300, 400, 500]
}

# Instantiate the Random Forest model
rf_class = RandomForestClassifier()

# Tune the model by finding the best hyperparameters
rf_grid_search = GridSearchCV(estimator=rf_class, scoring=['roc_auc'],
                              refit='roc_auc', param_grid=param_grid, cv=10,
                              n_jobs=-1, verbose=10)

# Fit the Random Forest model based on the best parameters
rf_grid_search.fit(X_train, y_train)

# Predict the target variable in the form of probabilities [p0, p1]
# where p0: Probability to belong to class 'no'  (i.e. not subscribed)
#       p1: Probability to belong to class 'yes' (i.e. subscribed)
rf_pred_probs = rf_grid_search.predict_proba(X_test)

# Assign the above probabilities to the corresponding class ('no', 'yes')
rf_y_pred = rf_grid_search.predict(X_test)

# Evaluate the model by using Recall/Precission:
mcf.getModelEvaluationMetrics(classifier=rf_grid_search,
                              model_name='Random Forest',
                              x_test=X_test, y_test=y_test,
                              y_predicted=rf_y_pred,
                              plot_confusion_matrix=True)

# Evaluate the model by using ROC Curve:
mcf.createROCAnalysis(classifier=rf_grid_search,
                      model_name='Random Forest', y_test=y_test,
                      pred_probs=rf_pred_probs, plot_ROC_Curve=True)

rf_grid_search.best_estimator_
'''
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=20, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=9, min_samples_split=20,
            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
'''

# Find the feature importance based on Gini criterion
feature_importance = {}
best_estimator_fi = rf_grid_search.best_estimator_.feature_importances_

for feature, importance in zip(X_train.columns, best_estimator_fi):
    feature_importance[feature] = importance

importances = pd.DataFrame.from_dict(feature_importance,
                                     orient='index').rename(
                                             columns={0: 'Gini Score'})

importances = importances.sort_values(by='Gini Score', ascending=False)

# Plot for feature importance
plt.figure(figsize=(14, 12))
sns.barplot(x=importances.index[0:10],
            y=importances['Gini Score'].iloc[0:10], palette='muted')
plt.title(f'Importance for the Top 10 Features (Gini criterion) ',
          fontweight='bold')
plt.grid(True, alpha=0.1, color='black')
plt.show()


# Pick only 5 variables for the second model
pout_ls = [col for col in X_train.columns if col.startswith('poutcome')]
five_features_ls = ['nr.employed', 'age', 'pdays_bucket', 'campaign'] + pout_ls
X_train_reduced = X_train[five_features_ls]
X_test_reduced = X_test[five_features_ls]


# Model B: Logistic Regression with best 5 variables
bal_w = ['balanced', {0: 1, 1: 9.0}, {0: 0.1, 1: 1.0}]

param_grid = {
    'solver': ['lbfgs', 'saga', 'sag'],
    'C': np.arange(0.1, 20.1, 0.1),
    'penalty': ['l2'],
    'class_weight': bal_w
}

# Instantiate the Logistic Regression Model
log_reg = LogisticRegression()

# Tune the model by finding the best hyperparameters
logreg_grid_search = GridSearchCV(estimator=log_reg, scoring=['roc_auc'],
                                  refit='roc_auc', param_grid=param_grid,
                                  cv=10, n_jobs=-1, verbose=10)

# Fit the model and find the optimal parameters
logreg_grid_search.fit(X_train_reduced, y_train)

logreg_pred_probs = logreg_grid_search.predict_proba(X_test_reduced)

# Assign the above probabilities to the corresponding class ('no', 'yes')
logreg_y_pred = logreg_grid_search.predict(X_test_reduced)


# Evaluate the model by using Recall/Precission:
mcf.getModelEvaluationMetrics(classifier=logreg_grid_search,
                              model_name='Logistic Regression',
                              x_test=X_test_reduced, y_test=y_test,
                              y_predicted=logreg_y_pred,
                              plot_confusion_matrix=True)

# Evaluate the model by using ROC Curve:
mcf.createROCAnalysis(classifier=logreg_grid_search,
                      model_name='Logistic Regression', y_test=y_test,
                      pred_probs=logreg_pred_probs, plot_ROC_Curve=True)

logreg_grid_search.best_estimator_
'''
LogisticRegression(C=5.3, class_weight={0: 1, 1: 9.0}, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='warn', n_jobs=None, penalty='l2', random_state=None,
          solver='saga', tol=0.0001, verbose=0, warm_start=False)
'''
