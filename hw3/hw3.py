from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import KFold
import graphviz
import os

os.system("rm iris*")
iris = load_iris()

x = iris.data
y = iris.target

# USE GridSearchCV function
parameters = {'max_depth' : range(3,20)}
clf = GridSearchCV(tree.DecisionTreeClassifier(),parameters)
clf.fit(X=x,y=y)
tree_model = clf.best_estimator_
print(clf.best_score_,clf.best_params_)

#USE cross_val_score function in loop
# fold = []
# depth = []
# for i in range(3,20):
#     clf = tree.DecisionTreeClassifier(max_depth=i)
#     scores = cross_val_score(estimator=clf, X=x, y=y, cv=7)
#     depth.append(scores.mean())
#     fold.append(i)
# print(depth.index(max(depth))+3, max(depth),"\n")
#
# for i in range(len(fold)):
#     print(fold[i],depth[i])

kf = KFold(n_splits= clf.best_params_['max_depth'],shuffle=True)
# print(kf.get_n_splits(x))
count = 1

for train_index, test_index in kf.split(x):
    result = 0
    print("TRAIN:", train_index, "TEST:",test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("iris"+ str(count))
    count += 1
    predict = clf.predict(X_test)
    print(predict)
    for i in range(len(y_test)):
        if (y_test[i] == predict[i]):
            result += 1

    print("Accuracy: {:.2f}%".format(result/len(y_test)*100))
