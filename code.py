import  sklearn
import numpy as np
import pandas as pd
#import tensorflow as tf
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import confusion_matrix

###############################################################

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

###############################################################
# FUNCTIONS

def metrics(Y_test, y_pred):
    print('---------------------------------------------------------------')
    print('Accuracy: %.2f' % accuracy_score(Y_test,   y_pred) )
    confmat = confusion_matrix(y_true=Y_test, y_pred=y_pred)
    print("confusion matrix")
    print(confmat)
    print('Precision: %.3f' % precision_score(y_true=Y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=Y_test, y_pred=y_pred))
    print('F1-measure: %.3f' % f1_score(y_true=Y_test, y_pred=y_pred))
    print('\n\n\n\n')


def convert_numeric(x):
    d = []
    for y in x:
        if y=='na':
            d.append(np.nan)
        else:
            d.append(float(y))
    return d



# Counting null values in all the columns
def count_null(x):
    count = 0
    for i in x:
        if i=='na':
            count+=1
    return count



# list reverse
def Reverse(lst):
    return [element for element in reversed(lst)]


###############################################################
## LOAD DATA

df_train = pd.read_csv('train.csv', header=None)
df_test = pd.read_csv('test.csv', header=None)

df_train = df_train.drop(df_train.index[[0]])
df_test = df_test.drop(df_train.index[[0]])



# separating labels into different variable
y = df_train.iloc[:, :1]


# deleting labels from the train set
df_train = df_train.drop([0], axis=1)


# counting the null values 
columns = df_train.columns
count = {}
for col in columns:
    count[col] = count_null(df_train[col].values)

sorted_count = sorted(count.items(), key=lambda x : x[1])

# sorting the columns with highest null values
sorted_count = Reverse(sorted_count)
countin = {}
for x in sorted_count:
    countin[x[0]] = x[1]

# dropping the columns with null values
df_train = df_train.drop([79,78,77,76,113,2,75,74,72,110,94,93,92,4,132,131,130,129,128,127,126,125,124,157,99,158,98,20,87], axis=1)

# converting all the labels into numeric
y_class = []
for y in y.values:
    if y=='neg':
        y_class.append(0)
    else:
        y_class.append(1)

y_train = pd.DataFrame(y_class)
x_train = df_train

# converting all the values into float64
cl = x_train.columns
num_data = pd.DataFrame()
for d in range(0,len(cl)):
        num_data[cl[d]] = convert_numeric(x_train[cl[d]].values)

X_train = num_data


#----------------------------------------------------------------------
# test set

# separating labels into different variable
y = df_test.iloc[:, :1]

# deleting labels from the train set
df_test = df_test.drop([0], axis=0)
y = df_test.iloc[:, :1]
del df_test[0]

# counting the null values 
columns = df_test.columns
count = {}
for col in columns:
    count[col] = count_null(df_test[col].values)

sorted_count = sorted(count.items(), key=lambda x : x[1])

sorted_count = Reverse(sorted_count)
countin = {}
for x in sorted_count:
    countin[x[0]] = x[1]

# dropping the columns with null values
df_test = df_test.drop([79,78,77,76,113,2,75,74,72,110,94,93,92,4,132,131,130,129,128,127,126,125,124,157,99,158,98,20,87], axis=1)


# converting all the values into numeric
y_class = []
for y in y.values:
    if y=='neg':
        y_class.append(0)
    else:
        y_class.append(1)

y_test = pd.DataFrame(y_class)
x_test = df_test

cl = x_test.columns
num_data = pd.DataFrame()
for d in range(0,len(cl)):
        num_data[cl[d]] = convert_numeric(x_test[cl[d]].values)
X_test = num_data

#--------------------------------------------------------------------------
from sklearn.preprocessing import Imputer

# create imputer to replace missing values with mean
imp = Imputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train)
X_train_imp = imp.transform(X_train)


impt = Imputer(missing_values=np.nan, strategy='mean')
impt = impt.fit(X_test)
X_test_imp = impt.transform(X_train)

X_train, X_test, Y_train, Y_test = train_test_split(X_train_imp,y_train, test_size=0.20, random_state=42)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
#--------------------------------------------------------------------------


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',
                               n_estimators=10,
                               random_state=1,
                               n_jobs=2)
forest.fit(X_train, Y_train.values.ravel())
y_pred = forest.predict(X_test)
print("Random Forest Classifier")
metrics(Y_test = Y_test, y_pred=y_pred)

#-------------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy',
                               max_depth=30, random_state=0)
tree.fit(X_train, Y_train)
y_pred = tree.predict(X_test)
print('Decison Tree Classifier')
metrics(Y_test = Y_test, y_pred=y_pred)

#------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, Y_train.values.ravel())
y_pred = knn.predict(X_test_std)
print('K-Nearest Neighbors')
metrics(Y_test = Y_test, y_pred=y_pred)

#------------------------------------------------------------------------
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=0.0010, C=32, probability=True)
svm.fit(X_train_std, Y_train.values.ravel())
y_pred = svm.predict(X_test_std)
print('Support Vector Machine')
metrics(Y_test = Y_test, y_pred=y_pred)

#------------------------------------------------------------------------






