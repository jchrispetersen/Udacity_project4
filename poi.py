
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list1 = ['poi',
 'salary',
 'to_messages',
 'deferral_payments',
 'total_payments',
 'loan_advances',
 'bonus',
 'restricted_stock_deferred',
 'deferred_income',
 'total_stock_value',
 'expenses',
 'exercised_stock_options',
 'from_messages',
 'other',
 'long_term_incentive',
 'shared_receipt_with_poi',
 'director_fees'] # You will need to use more features

features_list2 = ['poi',
 'salary',
 'to_messages',
 'deferral_payments',
 'total_payments',
 'loan_advances',
 'bonus',
 'restricted_stock_deferred',
 'deferred_income',
 'total_stock_value',
 'expenses',
 'exercised_stock_options',
 'from_messages',
 'other',
 'long_term_incentive',
 'shared_receipt_with_poi',
 'director_fees',
 'has_tpayment',
 'has_tsv',
 'has_expenses',
 'has_other',
 'fraction_from_poi',
 'fraction_to_poi',
 'total_beso',
 'num_millions']
### Load the dictionary containing the dataset
with open("final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('LOCKHART EUGENE E')
### Task 3: Create new feature(s)

for name in data_dict:
    data = data_dict[name]
    payment = data["total_payments"]
    if payment != 'NaN':
        has_tpayment = 1
    else:
        has_tpayment = 0
    data_dict[name]["has_tpayment"] = has_tpayment

for name in data_dict:
    data = data_dict[name]
    payment = data["total_stock_value"]
    if payment != 'NaN':
        has_tsv = 1
    else:
        has_tsv = 0
    data_dict[name]["has_tsv"] = has_tsv

for name in data_dict:
    data = data_dict[name]
    payment = data["expenses"]
    if payment != 'NaN':
        has_expenses = 1
    else:
        has_expenses = 0
    data_dict[name]["has_expenses"] = has_expenses

for name in data_dict:
    data = data_dict[name]
    payment = data["other"]
    if payment != 'NaN':
        has_other = 1
    else:
        has_other = 0
    data_dict[name]["has_other"] = has_other


def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = float(poi_messages)/all_messages


    return fraction


for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )

    data_dict[name]["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )

    data_dict[name]["fraction_to_poi"] = fraction_to_poi

for name in data_dict:
    data_point = data_dict[name]

    bonus = data_point['bonus']
    if bonus == 'NaN':
        bonus = 0.0
    options = data_point['exercised_stock_options']
    if options == 'NaN':
        options = 0.0
    total = bonus+options

    data_dict[name]['total_beso'] = total

for name in data_dict:
    data_point = data_dict[name]

    total_payments = data_point['total_payments']
    if total_payments == 'NaN':
        total_payments = 0.0
    total_stock = data_point['total_stock_value']
    if total_stock == 'NaN':
        total_stock = 0.0
    total = (total_payments + total_stock)/1000000

    data_dict[name]['num_millions'] = total

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data1 = featureFormat(my_dataset, features_list1, sort_keys = True)
labels1, features1 = targetFeatureSplit(data1)
data2 = featureFormat(my_dataset, features_list2, sort_keys = True)
labels2, features2 = targetFeatureSplit(data2)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
gnb_clf = GaussianNB()

gnb_clf.fit(features1, labels1)
print("Accuracy without new features:", gnb_clf.score(features1, labels1))
gnb_clf.fit(features2, labels2)
print("Accuracy with new features:", gnb_clf.score(features2, labels2))
gnb_clf.fit(features2, labels2)


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

skb = SelectKBest(k=11)
selected_features = skb.fit_transform(features2,labels2)
features_selected=[features_list2[i+1] for i in skb.get_support(indices=True)]

print('Features selected by SelectKBest:')
#print(selected_features)
print(features_selected)
features_list = features_selected

# Example starting point. Try investigating other evaluation techniques!
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

features_list.insert(0,'poi')
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state=42)

#GaussianNB
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()
clf1.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)
clf = GaussianNB()

#RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
#clf = RandomForestClassifier(n_estimators=100, min_samples_split=3)
clf2 = RandomForestClassifier(n_estimators=30, min_samples_split=5)
parameters = {'min_samples_split':[2,3,4,5,6], 'n_estimators': [10,20,30,40, 50, 60, 70, 80, 90, 100]}
random = RandomForestClassifier()
#clf = model_selection.GridSearchCV(random, parameters)
clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)

#adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection
ada = AdaBoostClassifier()
parameters = {'n_estimators':[10,50,100], 'random_state': [None, 0, 42, 138]}
clf3 = model_selection.GridSearchCV(ada, parameters)
clf3 = AdaBoostClassifier(n_estimators=50, random_state=138)
clf3.fit(features_train, labels_train)
pred3 = clf3.predict(features_test)

#DecisionTree
from sklearn import model_selection, tree
parameters = {'min_samples_split':[2,3,4,5,6,7,8,9], 'min_samples_leaf':[1,2,3], 'random_state':[None, 0, 42] }
clf4 = model_selection.GridSearchCV(tree, parameters)
clf4 = tree.DecisionTreeClassifier(min_samples_split = 5, random_state=42)
clf4 = clf4.fit(features_train, labels_train)
pred4 = clf4.predict(features_test)


print("GaussianNB")
test_classifier(clf1, my_dataset, features_list)
print("RandomForrest")
test_classifier(clf2, my_dataset, features_list)
print("adaBoost")
test_classifier(clf3, my_dataset, features_list)
print("DecisionTree")
test_classifier(clf4, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)
