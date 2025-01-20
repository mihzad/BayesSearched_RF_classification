import os

import numpy as np
import sklearn.svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import time
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data 2.csv")
#print (df.head(5000).to_string())

X = df.drop(columns=["p", "target"])
y = df['target']

#unique_classes = np.unique(y)
#print("unique classes: ", unique_classes)

X = X.drop(columns=["f10","f12","f13","f14","f15","f16"]) #corr redundant features
#f6 was not touched because corr = 0.82 < 0.9


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43, stratify=y)

#corr_matrix = X.corr()
#plt.figure(figsize=(10, 8))
#sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
#plt.title('Correlation Matrix Heatmap')
#plt.show()


#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train.values)
#X_test = scaler.transform(X_test.values)
#pca = PCA(n_components=0.95, svd_solver='full')
#pca.fit(X_train)
#X_train = np.array(pca.transform(X_train))
#X_test = np.array(pca.transform(X_test))


model = RandomForestClassifier(n_estimators=50, max_features='sqrt', max_depth=38, min_samples_split=4,min_samples_leaf=1, max_samples=0.44,
                                  criterion='entropy', random_state=43,
                                  n_jobs=-1, class_weight="balanced")

param_space = {
    'n_estimators': (30, 120),
    'max_features': (2, 7),
    'max_depth': (20, 50),
    'min_samples_split': (2, 12),
    'min_samples_leaf': (1, 5),
    'max_samples': (0.33, 1)
}
#custom_scorer = make_scorer(f1_score, greater_is_better=True, pos_label='yes', average='binary')
#bayes_search = BayesSearchCV(model, param_space, scoring=custom_scorer, n_iter=100, cv=4, random_state=42, n_jobs=1, n_points=1)
print("Bayesing the RF...")
start = time.time()
#bayes_search.fit(X_train, y_train)
model.fit(X_train, y_train)
end = time.time()
print (f"done. elapsed training time: {end-start} s\n")

#model = bayes_search.best_estimator_

print("Random Forest Model Information:")
print(f"Number of Trees (n_estimators): {model.n_estimators}")

tree_depths = [estimator.tree_.max_depth for estimator in model.estimators_]
print(f"Average Tree Depth: {sum(tree_depths)/len(tree_depths):.2f}")
print(f"Maximum Tree Depth: {max(tree_depths)}")

print("Feature Importances:")
for idx, importance in enumerate(model.feature_importances_):
    print(f"Feature {idx}: {importance:.4f}")

print("\nHyperparameters:")
print(model.get_params())

print(f"Number of Features Used: {model.n_features_in_}")

#=======TESTING PART=========
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


