# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```
Developed by:Adhithya Perumal.D
Registor No :212222230007
```
```
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.datasets import load_boston
boston = load_boston()

print(boston['DESCR'])

import pandas as pd
df = pd.DataFrame(boston['data'] )
df.head()

df.columns = boston['feature_names']
df.head()

df['PRICE']= boston['target']
df.head()

df.info()

plt.figure(figsize=(10, 8))
sns.distplot(df['PRICE'], rug=True)
plt.show()

#FILTER METHODS
X=df.drop("PRICE",1)
y=df["PRICE"]

from sklearn.feature_selection import SelectKBest, chi2
X, y = load_boston(return_X_y=True)
X.shape

#1.VARIANCE THRESHOLD
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit_transform(X)

#2.INFORMATION GAIN/MUTUAL INFORMATION
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y);
mi = pd.Series(mi)
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))

#3.SELECTKBEST METHOD
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest,SelectPercentile
skb = SelectKBest(score_func=f_classif, k=2) 
X_data_new = skb.fit_transform(X, y)
print('Number of features before feature selection: {}'.format(X.shape[1]))
print('Number of features after feature selection: {}'.format(X_data_new.shape[1]))

#4.CORRELATION COEFFICIENT
cor=df.corr()
sns.heatmap(cor,annot=True)

#5.MEAN ABSOLUTE DIFFERENCE
mad=np.sum(np.abs(X-np.mean(X,axis=0)),axis=0)/X.shape[0]
plt.bar(np.arange(X.shape[1]),mad,color='teal')

#Processing data into array type.
from sklearn import preprocessing
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
print(y_transformed)

#6.CHI SQUARE TEST
X = X.astype(int)
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(X, y_transformed)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])

#7.SELECT PERCENTILE METHOD
X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y_transformed)
X_new.shape

#WRAPPER METHOD
#1.FORWARD FEATURE SELECTION

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
sfs = SFS(LinearRegression(),
          k_features=10,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)
sfs.fit(X, y)
sfs.k_feature_names_

#2.BACKWARD FEATURE ELIMINATION

sbs = SFS(LinearRegression(),
         k_features=10,
         forward=False,
         floating=False,
         cv=0)
sbs.fit(X, y)
sbs.k_feature_names_

#3.BI-DIRECTIONAL ELIMINATION

sffs = SFS(LinearRegression(),
         k_features=(3,7),
         forward=True,
         floating=True,
         cv=0)
sffs.fit(X, y)
sffs.k_feature_names_

#4.RECURSIVE FEATURE SELECTION
from sklearn.feature_selection import RFE
lr=LinearRegression()
rfe=RFE(lr,n_features_to_select=7)
rfe.fit(X, y)
print(X.shape, y.shape)
rfe.transform(X)
rfe.get_params(deep=True)
rfe.support_
rfe.ranking_

#EMBEDDED METHOD

#1.RANDOM FOREST IMPORTANCE
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X,y_transformed)
importances=model.feature_importances_

final_df=pd.DataFrame({"Features":pd.DataFrame(X).columns,"Importances":importances})
final_df.set_index("Importances")
final_df=final_df.sort_values("Importances")
final_df.plot.bar(color="teal")
```
## OUTPUT:

![1](https://user-images.githubusercontent.com/118707079/234245250-16638c13-3c9f-4d64-9d5e-c555649a258f.png)

## Analyzing the boston dataset:

![2](https://user-images.githubusercontent.com/118707079/234245460-3e362d40-ddbd-4057-871a-7111998784f7.png)

![3](https://user-images.githubusercontent.com/118707079/234245680-a57e8f1c-f06a-404a-aa91-a94f4f607138.png)

![4](https://user-images.githubusercontent.com/118707079/234245806-30af5d8e-3e4b-4ce2-96c7-df7c88d2f02f.png)

## Analyzing dataset using Distplot:

![5 1](https://user-images.githubusercontent.com/118707079/234246634-aa912e56-890c-47ed-b530-ba31d73eac17.png)
![5 2](https://user-images.githubusercontent.com/118707079/234246633-e071212d-530e-4bd7-ad8e-fe1a623dc275.png)

## Filter Methods:
Variance Threshold:

![6](https://user-images.githubusercontent.com/118707079/234246827-e194e7cd-b431-4e6f-b901-4d6d210c0f2a.png)

## Information Gain:

![7](https://user-images.githubusercontent.com/118707079/234247277-aa1603d4-7cc1-4de1-88e3-8cdc87bbfedc.png)

## SelectKBest Model:

![8](https://user-images.githubusercontent.com/118707079/234247499-adb36414-dc08-4960-86b9-c827a0b70971.png)

## Correlation Coefficient:

![9](https://user-images.githubusercontent.com/118707079/234247618-fe85c7e0-1726-4746-8417-56069c43da5c.png)

## Mean Absolute difference:

![10](https://user-images.githubusercontent.com/118707079/234247897-b32f9bd3-d1db-4e49-83e1-75e30a7af287.png)

## Chi Square Test:

![11](https://user-images.githubusercontent.com/118707079/234248021-923c053c-561b-4767-a77f-c6edb008cae2.png)
![12](https://user-images.githubusercontent.com/118707079/234248255-7f1e81a4-c4ac-435c-8b15-6264918ecbce.png)

## SelectPercentile Method:

![13](https://user-images.githubusercontent.com/118707079/234248408-a32511e0-e807-4b4e-a062-65de74162c66.png)

## Wrapper Methods:
Forward Feature Selection:

![14](https://user-images.githubusercontent.com/118707079/234249756-9d1ad92d-a34e-487d-9345-47f6498139b9.png)

Backward Feature Selection:

![15](https://user-images.githubusercontent.com/118707079/234249890-32f4a579-e6a8-484c-a57b-47bba97eb891.png)

## Bi-Directional Elimination:

![16](https://user-images.githubusercontent.com/118707079/234250023-3f831d59-d840-4942-a673-1aceee5227cf.png)

## Recursive Feature Selection:

![17](https://user-images.githubusercontent.com/118707079/234250134-628cc4ab-bcd2-45ae-b204-7581fbcfbdc6.png)

## Embedded Methods:
Random Forest Importance:

![18](https://user-images.githubusercontent.com/118707079/234250409-0b18ba51-d309-4740-b241-db907efd72c2.png)

## RESULT:

Hence various feature selection techniques are applied to the given data set successfully and saved the data into a file.















