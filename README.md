# Binary Classification of Direct Marketing Campaign Subscriptions: A Logistic Regression \& Random Forest Approach

Exploratory data analysis (EDA) and development of classification algorithms (Logistic Regression, Random Forest) to predict clients that are most likely to subscribe to a bank's product, as a result of marketing campaigns.


## Project Description

Purpose of this project is to analyze a dataset containing information about marketing campaigns that were conducted via phone calls from a Portuguese banking institution to their clients. The main goal of these campaigns was to prompt their clients to subscribe for a specific financial product of the bank (term deposit). After each call was conducted, the client had to inform the institution about their intention of either subscribing to the product (indicating a successful campaign) or not (unsucessful campaign).

Our main task in this project is to create effective machine learning algorithms that are able to predict the probability of a client subscribing to the bank's product. We should note that, even though we are talking about calculating probabilites, we will create classification algorithms - meaning that the final output of our models will be a binary result indicating if the client subscribed ('yes') to the product or not ('no').

The dataset has 41188 rows (instances of calls to clients) and 21 columns (variables) which are describing certain aspects of the call. Please note that there are cases where the same client was contacted multiple times - something that practically doesn't affect our analysis as each call will be considered independent from each other, even if the client is the same.


<img src="https://github.com/gpsyrou/Binary-Classification-on-Bank-Marketing-Campaigns/blob/master/project_picture.jpg" style="vertical-align:middle;margin:0px 0px">

Useful Links:
1. https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
2. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
3. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
4. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
5. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
6. https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/

The project's introduction picture has been taken from <a href="https://slideplayer.com/slide/15747121/">here</a>.
