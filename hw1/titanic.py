#!/usr/bin/python
import sys, math
import numpy as np
import matplotlib
matplotlib.use('TKAgg') #MAC에서 그래프 잘보이게하는 명령어
import matplotlib.pyplot as plt
import pandas as pd

def extract_familyname(row):
    return row['name'].split(',')[0]

def import_titanic_csv(filename):
    print('')
    print('importing file %s' % filename)
    result_df = pd.read_csv(filename)

    print('\ttotal number of rows in training set: %d' % len(result_df.index))
    for key in ['age', 'sex', 'pclass', 'embarked', 'fare']:
        result_df = result_df[result_df[key].notnull()]
        print('\tremoved rows with no %8s: %d rows left' % (key, len(result_df.index)))

    age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    age_cutline = range(0, 81, 10)
    result_df['age_group'] = pd.cut(result_df.age, age_cutline, right=False, labels=age_labels)

    fare_labels = ['-019', '-039', '-059', '-079', '-099', '-199', '-299', '-600']
    fare_cutline = [0,20,40,60,80,100,200,300,1000]
    result_df['fare_group'] = pd.cut(result_df.fare, fare_cutline, right=False, labels=fare_labels)

    result_df['family_size'] = result_df['sibsp'] + result_df['parch']

    result_df['family_name'] = result_df.apply(extract_familyname, axis=1)
    print('')
    return result_df

train_df = import_titanic_csv('titanic_train.csv')

sgroup = train_df.groupby('sex').survived
print('-- dataframe --')
print(train_df)
print('')
print('-- group keys --')
print(sorted(sgroup.groups.keys()))
print('-- group count --')
print(sgroup.count())
print('-- group sum --')
print(sgroup.sum())
print('-- group mean --')
print(sgroup.mean())
print('')

# raw_input()

def graph(df, feature_column, interest_column, figure_index=1, show_now=True, color_true='g', label_true='Survivors', label_false='Non-Survivors'):

    mygroup = df.groupby(feature_column)[interest_column]
    categories =  sorted(mygroup.groups.keys())
    total_count = mygroup.count().values
    true_count = mygroup.sum().values
    true_ratio = mygroup.mean().values
    false_count = np.subtract(total_count, true_count)
    false_ratio = np.subtract(1.0, true_ratio)


    plt.figure(figure_index, figsize=(15,10))
    plt.subplot(121)
    ax1 = plt.gca()

    ax1.bar(range(len(categories)), true_count, label=label_true, alpha=0.5, color=color_true)
    ax1.bar(range(len(categories)), false_count, bottom=true_count, label=label_false, alpha=0.5, color='r')
    plt.sca(ax1)
    plt.xticks(np.add(range(len(categories)), 0.4), categories)
    ax1.set_ylabel("Count")
    # ax1.set_xlabel(feature_column)
    ax1.set_title("Count of %s by %s" % (label_true, feature_column),fontsize=14)
    plt.legend(loc='upper left')

    plt.subplot(122)
    ax2 = plt.gca()
    # plot chart for percentage of survivors by class
    ax2.bar(range(len(categories)), true_ratio, alpha=0.5, color=color_true)
    ax2.bar(range(len(categories)), false_ratio, bottom=true_ratio, alpha=0.5, color='r')
    plt.sca(ax2)
    plt.xticks(np.add(range(len(categories)), 0.4), categories)
    ax2.set_ylabel("Percentage")
    # ax2.set_xlabel(feature_column)
    ax2.set_title("%% of %s by %s" % (label_true, feature_column),fontsize=14)
    if show_now:
        plt.show()
    return

figure_index = 1
for key in ['sex', 'age_group', 'family_size', 'pclass', 'embarked', 'fare_group']:
    graph(train_df, key, 'survived', figure_index=figure_index, show_now=False)
    figure_index += 1

plt.draw()

def guess_survival(row):
    # rules induced
    '''
    family size 1-3 survives
    embarked at c survives
    pclass 1 survives 2 neutral
    fare over 80 survies 40-80 neutral
    age below 10 survives, other neutral, over 70 dies
    male dies, female lives

    '''
    guess = 0.0
    coefficients = {
        'family_size' : 1.0,
        'embarked' : 1.0,
        'pclass' : 1.0,
        'fare' : 1.0,
        'age' : 1.0,
        'sex' : 5.0 # 5.0, 20.0
    }

    if row['family_size'] >= 1 or row['family_size'] <= 3:
        guess += 1.0 * coefficients['family_size']
    elif False:
        guess += 0.5 * coefficients['family_size']

    if row['embarked'] == 'C':
        guess += 1.0 * coefficients['embarked']
    elif False:
        guess += 0.5 * coefficients['embarked']

    if row['pclass'] == 1:
        guess += 1.0 * coefficients['pclass']
    elif row['pclass'] == 2:
        guess += 0.5 * coefficients['pclass']

    if row['fare'] >= 80:
        guess += 1.0 * coefficients['fare']
    elif row['fare'] >= 40:
        guess += 0.5 * coefficients['fare']

    if row['age'] <= 10:
        guess += 1.0 * coefficients['age']
    elif row['fare'] < 70:
        guess += 0.5 * coefficients['age']

    if row['sex'] == 'female':
        guess += 1.0 * coefficients['sex']
    elif False:
        guess += 0.5 * coefficients['sex']

    if guess >= sum(coefficients.values()) * 0.5:
        return 1
    else:
        return 0

train_df['guess'] = train_df.apply(guess_survival, axis=1)

def check_survival(row):
    if row['survived'] == row['guess']:
        return 1
    else:
        return 0

train_df['guess_result'] = train_df.apply(check_survival, axis=1)

# for key in ['sex', 'age_group', 'family_size', 'pclass', 'embarked', 'fare_group']:
#    graph(train_df, key, 'guess_result', figure_index=figure_index, show_now=False, label_true='Correct guess', label_false='Incorrect guess', color_true='b')
#    figure_index += 1

test_df = import_titanic_csv('titanic_full.csv')
test_df = test_df.loc[~test_df['name'].isin(train_df.name)]
print('\tremoved rows with %11s: %d rows left\n' % ('duplicates', len(test_df.index)))

print('TRAINING SET ACCURACY: %f' % train_df['guess_result'].mean())

test_df['guess'] = test_df.apply(guess_survival, axis=1)
test_df['guess_result'] = test_df.apply(check_survival, axis=1)
print('TEST SET ACCURACY: %f\n' % test_df['guess_result'].mean())


from sklearn import tree
from sklearn import preprocessing

# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()
# Convert Sex variable to numeric
encoded_sex = label_encoder.fit_transform(train_df.sex)
encoded_sex_test = label_encoder.fit_transform(test_df.sex)
# Initialize model
tree_model = tree.DecisionTreeClassifier()

print('features: ' + str(["sex"]))
# Train the model
tree_model.fit(X = pd.DataFrame(encoded_sex), y = train_df.survived)
# Save tree as dot file
with open("tree1.dot", 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=["sex"], out_file=f)
# Get survival probability
predictors = tree_model.predict_proba(X = pd.DataFrame(encoded_sex))
# print predictors
print(pd.crosstab(predictors[:,0], train_df.sex))
# Make test set predictions
test_features = pd.DataFrame(encoded_sex)
train_df['guess'] = tree_model.predict(X=test_features)
train_df['guess_result'] = train_df.apply(check_survival, axis=1)
print('TRAINING SET ACCURACY: %f' % test_df['guess_result'].mean())
test_features = pd.DataFrame(encoded_sex_test)
test_df['guess'] = tree_model.predict(X=test_features)
test_df['guess_result'] = test_df.apply(check_survival, axis=1)
print('TEST     SET ACCURACY: %f\n' % test_df['guess_result'].mean())


print('features: ' + str(["sex", "pclass"]))
# Make data frame of predictors
predictors = pd.DataFrame([encoded_sex, train_df.pclass]).T
# Train the model
tree_model.fit(X = predictors, y = train_df.survived)
# Save tree as dot file
with open("tree2.dot", 'w') as f:
    f = tree.export_graphviz(tree_model,
                             feature_names=["sex", "pclass"],
                             out_file=f)
# Make test set predictions
test_features = pd.DataFrame([encoded_sex,train_df.pclass]).T
train_df['guess'] = tree_model.predict(X=test_features)
train_df['guess_result'] = train_df.apply(check_survival, axis=1)
print('TRAINING SET ACCURACY: %f' % test_df['guess_result'].mean())
test_features = pd.DataFrame([encoded_sex_test,test_df.pclass]).T
test_df['guess'] = tree_model.predict(X=test_features)
test_df['guess_result'] = test_df.apply(check_survival, axis=1)
print('TEST     SET ACCURACY: %f\n' % test_df['guess_result'].mean())



print('features: ' + str(["sex", "pclass", "fare"]))
# Make data frame of predictors
predictors = pd.DataFrame([encoded_sex, train_df.pclass, train_df.fare]).T
# Train the model
tree_model.fit(X = predictors, y = train_df.survived)
# Save tree as dot file
with open("tree3.dot", 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=["sex", "pclass", "fare"], out_file=f)
# Make test set predictions
test_features = pd.DataFrame([encoded_sex, train_df.pclass, train_df.fare]).T
train_df['guess'] = tree_model.predict(X=test_features)
train_df['guess_result'] = train_df.apply(check_survival, axis=1)
print('TRAINING SET ACCURACY: %f' % test_df['guess_result'].mean())
test_features = pd.DataFrame([encoded_sex_test,test_df.pclass, test_df.fare]).T
test_df['guess'] = tree_model.predict(X=test_features)
test_df['guess_result'] = test_df.apply(check_survival, axis=1)
print('TEST     SET ACCURACY: %f\n' % test_df['guess_result'].mean())



# Initialize model with maximum tree depth set to 8
tree_model = tree.DecisionTreeClassifier(max_depth = 8)
print('setting max tree depth to 8\n')

print('features: ' + str(["sex", "pclass", "fare", "age"]))
# Make data frame of predictors
predictors = pd.DataFrame([encoded_sex, train_df.pclass, train_df.fare, train_df.age]).T
# Train the model
tree_model.fit(X = predictors, y = train_df.survived)
# Save tree as dot file
with open("tree4.dot", 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=["sex", "pclass", "fare", "age"], out_file=f)
# Make test set predictions
test_features = pd.DataFrame([encoded_sex,train_df.pclass, train_df.fare, train_df.age]).T
train_df['guess'] = tree_model.predict(X=test_features)
train_df['guess_result'] = train_df.apply(check_survival, axis=1)
print('TRAINING SET ACCURACY: %f' % test_df['guess_result'].mean())
test_features = pd.DataFrame([encoded_sex_test,test_df.pclass, test_df.fare, test_df.age]).T
test_df['guess'] = tree_model.predict(X=test_features)
test_df['guess_result'] = test_df.apply(check_survival, axis=1)
print('TEST     SET ACCURACY: %f\n' % test_df['guess_result'].mean())



# Convert Sex variable to numeric
encoded_embarked = label_encoder.fit_transform(train_df.embarked)
encoded_embarked_test = label_encoder.fit_transform(test_df.embarked)

print('features: ' + str(["sex", "pclass", "fare", "age", "embarked"]))
# Make data frame of predictors
predictors = pd.DataFrame([encoded_sex, train_df.pclass, train_df.fare, train_df.age, encoded_embarked]).T
# Train the model
tree_model.fit(X = predictors, y = train_df.survived)
# Save tree as dot file
with open("tree5.dot", 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=["sex", "pclass", "fare", "age", "embarked"], out_file=f)
# Make test set predictions
test_features = pd.DataFrame([encoded_sex, train_df.pclass, train_df.fare, train_df.age, encoded_embarked]).T
train_df['guess'] = tree_model.predict(X=test_features)
train_df['guess_result'] = train_df.apply(check_survival, axis=1)
print('TRAINING SET ACCURACY: %f' % test_df['guess_result'].mean())
test_features = pd.DataFrame([encoded_sex_test,test_df.pclass, test_df.fare, test_df.age, encoded_embarked_test]).T
test_df['guess'] = tree_model.predict(X=test_features)
test_df['guess_result'] = test_df.apply(check_survival, axis=1)
print('TEST     SET ACCURACY: %f\n' % test_df['guess_result'].mean())



print('features: ' + str(["sex", "pclass", "fare", "age", "embarked", "family_size"]))
# Make data frame of predictors
predictors = pd.DataFrame([encoded_sex, train_df.pclass, train_df.fare, train_df.age, encoded_embarked, train_df.family_size]).T
# Train the model
tree_model.fit(X = predictors, y = train_df.survived)
# Save tree as dot file
with open("tree6.dot", 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=["sex", "pclass", "fare", "age", "embarked", "family_size"], out_file=f)
# Make test set predictions
test_features = pd.DataFrame([encoded_sex, train_df.pclass, train_df.fare, train_df.age, encoded_embarked, train_df.family_size]).T
train_df['guess'] = tree_model.predict(X=test_features)
train_df['guess_result'] = train_df.apply(check_survival, axis=1)
print('TRAINING SET ACCURACY: %f' % test_df['guess_result'].mean())
test_features = pd.DataFrame([encoded_sex_test,test_df.pclass, test_df.fare, test_df.age, encoded_embarked_test, test_df.family_size]).T
test_df['guess'] = tree_model.predict(X=test_features)
test_df['guess_result'] = test_df.apply(check_survival, axis=1)
print('TEST     SET ACCURACY: %f\n' % test_df['guess_result'].mean())


plt.show()

