###VOTING Ensemble
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

dataset = datasets.load_breast_cancer()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test= train_test_split(X, y,
test_size=0.2,
stratify=y,
random_state=1)

SEED=1

lr = LogisticRegression(random_state=SEED)
knn = KNeighborsClassifier(n_neighbors=27)
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:   
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('{:s} : {:.3f}'.format(clf_name, accuracy))


from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(estimators=classifiers)     
vc.fit(X_train, y_train)   
y_pred = vc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Voting Classifier: {:.3f}'.format(accuracy))

###Bagging
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,BaggingRegressor

dt = DecisionTreeClassifier(random_state=1)
bc = BaggingClassifier(base_estimator=dt, 
                       n_estimators=300, 
                       oob_score=True,
                       random_state=1)

bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
#OutOfBagScore
acc_oob = bc.oob_score_
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))