from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
num_splits = 10
num_epochs = 5
accuracy = 0.0
X, y = iris.data, iris.target

kf = KFold(n_splits=num_splits)
model = SGDClassifier()


for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for _ in range(5):
        model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    present_acc = accuracy_score(y_test, prediction)
    accuracy += present_acc

print("accuracy:", accuracy/num_splits)

