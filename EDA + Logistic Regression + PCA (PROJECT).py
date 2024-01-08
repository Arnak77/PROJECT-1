# EDA + Logistic Regression + PCA


## Import Python libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

## Import dataset
df=pd.read_csv(r"D:\NIT\JANUARY\3 JAN(logistic regression pra)\2nd,3rd\Project\adult.csv\adult.csv")

### Check shape of dataset
df.shape
### Preview dataset
df.head()
### View summary of dataframe
df.info()
df.dtypes
df.isnull().sum()

### Encode `?` as `NaNs`
df=df.replace('?',np.nan)

df.isnull().sum()
df.columns

### Impute missing values with mode
for col in ["workclass","occupation","native.country"]:
    df[col].fillna(df[col].mode()[0],inplace=True)

df.isnull().sum()

### Setting feature vector and target variable
X=df.drop(['income'],axis=1)
y=df['income']


## Split data into separate training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.20,random_state=0)


### Encode categorical variables
from sklearn.preprocessing import LabelEncoder
categorical_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

for feature in categorical_features:
    le = LabelEncoder()
    X_train[feature] = le.fit_transform(X_train[feature])
    X_test[feature] = le.transform(X_test[feature])


## Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

## Logistic Regression model with all features
from sklearn.linear_model import LogisticRegression 
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac) 
#Logistic Regression accuracy score with all the features: 0.8218

bias = regressor.score(X_train, y_train)
bias

variance = regressor.score(X_test, y_test)
variance

#Now, let's get to the PCA implementation.


from sklearn.decomposition import PCA
pca=PCA()
pca.fit_transform(X_train)
pca.explained_variance_ratio_


#####################################################

### Logistic Regression with first 13 features
df.columns
X=df.drop(['income','native.country'],axis=1)
y=df['income']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
for feature in categorical:
        le = LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)




from sklearn.linear_model import LogisticRegression 
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac) 
#Logistic Regression accuracy score with the first 13 features: 0.8213


from sklearn.decomposition import PCA
pca=PCA()
pca.fit_transform(X_train)
pca.explained_variance_ratio_
###############################################################


### Logistic Regression with first 12 features
X = df.drop(['income','native.country', 'hours.per.week'], axis=1)
y = df['income']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
for feature in categorical:
        le = LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)




from sklearn.linear_model import LogisticRegression 
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac) 

#Logistic Regression accuracy score with the first 12 features: 0.8227

from sklearn.decomposition import PCA
pca=PCA()
pca.fit_transform(X_train)
pca.explained_variance_ratio_



##################################################################



### Logistic Regression with first 11 features

df.columns
X=df.drop(['capital.loss', 'hours.per.week', 'native.country',
'income'],axis =1)

X.columns
y = df['income']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
for feature in categorical:
        le = LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)




from sklearn.linear_model import LogisticRegression 
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac) 
#### Logistic Regression with first 11 features


from sklearn.decomposition import PCA
pca=PCA()
pca.fit_transform(X_train)
pca.explained_variance_ratio_

################################################

X = df.drop(['income'], axis=1)
y = df['income']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical:
        le = LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

pca= PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
dim = np.argmax(cumsum >= 0.90) +1
print('The number of dimensions required to preserve 90% of variance is',dim)


#The number of dimensions required to preserve 90% of variance is 12


## Plot explained variance ratio with number of dimensions
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,14)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()



### Comment

#The above plot shows that almost 90% of variance is explained by the first 12 components.









