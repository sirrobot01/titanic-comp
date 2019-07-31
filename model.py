from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
X = np.loadtxt('train4', delimiter=',')
y = np.loadtxt('label.csv', delimiter=',')
to_pred = np.loadtxt('test4.csv', delimiter=',')
y = np.array(y)
X  = np.array(X[:, 1:])
to_pred = np.array(to_pred[:, 1:])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

###########################################################################
#MODELS

parameter_grid = [
                   {'alpha':[0.1,0.01,0.001,0.0001]}]

'''all_model = Pipeline([
    ('logre', LogisticRegression(solver='lbfgs', multi_class='auto')),
    ('svm',SVC( gamma='auto')),
    ('tree', DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)),
    ('random', RandomForestClassifier()),
    ('mlp', MLPClassifier(random_state=0))])
'''
model = GridSearchCV(estimator=MLPClassifier(random_state=0),#LogisticRegression(solver='lbfgs', multi_class='auto'),
                     param_grid=parameter_grid,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)

model.fit(X_train, y_train)
print(model.score(X_train,y_train))
print(model.score(X_test, y_test))
print(model.best_params_)
new_pred = model.predict(to_pred)
new_pred = pd.DataFrame(new_pred)
new_pred = new_pred.astype('int64')
new_pred.to_csv('tested5.csv', header=False, index_label=False, index=False)
#print(model.predict(X_test))'''

