classification problem with unbalanced binary dataset (99 to 1).
Solved by removing redundant (correlative) features and, after testing different models, random forest classifier. Used skopt.BayesSearchCV to find "basically good" RF model hyperparameters, then tuned them a bit further. Final model performed with 0.82 min precision and 0.77 min recall.
