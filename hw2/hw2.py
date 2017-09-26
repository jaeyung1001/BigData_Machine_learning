from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()
# print(iris)

#scatter[sepal length, sepal width]
X = iris.data[:, :2]
Z = iris.data[:, 2:4]
# print(Z)
# print(X)
y = iris.target

# print(X[:,0].min())

#sepal length&width
x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5

#petal length&width
z_min, z_max = Z[:,0].min()-0.5, Z[:,0].max()+0.5
y1_min, y1_max = Z[:,1].min()-0.5, Z[:,1].max()+0.5

plt.figure(1, figsize=(10,5))
plt.subplot(121)
plt.scatter(X[:,0],X[:,1],c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.title('Sepal length & width')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.subplot(122)
plt.scatter(Z[:,0],Z[:,1],c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.title('Petal length & width')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.xlim(z_min,z_max)
plt.ylim(y1_min,y1_max)
plt.show()
# plt.close()

# decision tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names = iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")



# predict test_vector
test_vector = [[5.0, 4.3, 2.0, 0.2],[6.4, 3.2, 5.3, 2.3]]
test_pd = pd.read_csv("cs-testing.csv", names=['species','sepal_length','sepal_width','petal_length','petal_width'])
# print(test_pd)
# print(test_pd.species)
list = []
# print(test_pd['sepal_length'][0])
for i in range(len(test_pd)):
    list.append([float(test_pd['sepal_length'][i]),float(test_pd['sepal_width'][i]),float(test_pd['petal_length'][i]),float(test_pd['petal_width'][i])])
# print(list[0])
# for test_vector in list:
# 	print(test_vector)
predict = clf.predict(list)
print(predict)

#accuracy
count = 0
for i in range(len(test_pd)):
    if(test_pd['species'][i]==predict[i]):
        count += 1

print("Accuracy: {}%".format(count/len(test_pd)*100))
