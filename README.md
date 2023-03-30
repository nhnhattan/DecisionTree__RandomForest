# CHURN MODELING

## I. Data Description

### 1. Reason for writing

- To be able to develop long-term, expand the scale, and compete directly with competitors, one of the factors that businesses always focus on is retaining loyal customers. Customer loyalty is the close relationship between sellers and buyers, trust in product quality, and service satisfaction. The common psychology of customers is that they all want businesses and service providers to meet their needs most thoroughly. That is not only the quality and price but also the consulting and care services.
- The banking industry is no exception. Today, many good banks can meet the needs of customers, so retaining customers is one of the most important things in the future. development as well as the competition of a bank with other banks.
- Therefore, it is necessary to analyze and determine the possibility that a customer can continue or stop using the bank's services.
- Because of the problem seeing the serious impact on the bank's revenue, our group has decided to choose this topic with the desire to study and apply data mining techniques to solve the posed problem.

### 2. Data Description

- Link dataset: https://www.kaggle.com/datasets/shubh0799/churn-modelling
- This data set contains detailed information about the bank's customers and a binary variable (Exited) that reflects whether the customer is likely to stop using the service (Close the account) or continue using the service at the bank.
- The data set includes 14 attribute columns and 10000 data rows.
<p align="center">
  <img src="https://cdn.discordapp.com/attachments/847349555703316512/1090991167115640862/image.png">
</p>

### 3. Descriptiontion of the problem

- **INPUT**:  
  o Data fields about customer's personal information (Gender, Age, Geography, EstimatedSalary, Tenure, NumofProducts). </br>
  o Data fields about customer’s bank account information (Balance, HascrCard, creditScore, IsActiveMember).
- **PROCESSING**: Using Decision Tree and Random Forest data mining algorithms. to predict whether or not customers will stop using the service at the bank
- **OUTPUT**: Do customers stop using services at the bank? 1 – Yes, 0 – No

## II. Data preprocessing

### 1. Data Visualization

- Install and import the necessary libraries

```python
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
```

- Read dataset.

```python
data = pd.read_csv('Churn_Modelling.csv')
```

- Check the number of data lines

```python
print('Số lượng dòng của dữ liệu: ', len(data))
```

- Check the data lines in the file.</br> <p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090993296094015488/image.png?width=1025&height=289">
</p>

- Check data type and data count statistics of each column

```python
data.info()
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090994095805169694/image.png">
</p>

- Check if the dataset has null data

```python
data.isna().sum()
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090994400127090738/image.png">
</p>

```python
import pandas as pd
import missingno as msno
msno.bar(data)
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090994468410368100/image.png">
</p>

- Check duplicate data

```python
sum(data.duplicated())
```

- Data statistics

```python
data.describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='RdPu').format('{:.2f}')
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090994879582195793/image.png">
</p>

- **EDA Model** <br>
  • Survey of similarity using Pearson formula, combined with the seaborn library to represent graphical visualization of Heatmap graph results.

```python
plt.title("features correlation matrics".title(),
        fontsize=20,weight="bold")

sns.heatmap(data.corr(),annot=True,cmap='PuBu',linewidths=0.2, vmin=-1, vmax=1,linecolor = 'black')
fig=plt.gcf()
fig.set_size_inches(10,8);
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090995316402167878/image.png?width=501&height=434">
</p>

• Statistics of customers leaving and staying

```python
print(data.Exited.value_counts())
ax = sns.countplot(data=data, x='Exited')
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090996033221300294/image.png">
</p>
=> The results show that the total number of departures (Exited = 1) is 2073, and the total number of stays (Exited = 0) is 7963
•	Statistics by Geography

```python
print(data.Geography.value_counts())
f,ax=plt.subplots(1,2,figsize=(18,8))
data['Geography'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Số lượng khách hàng của mỗi quốc gia')
ax[0].set_ylabel('count')

sns.countplot(data=data,x='Geography',hue='Exited',ax=ax[1])
ax[1].set_title('Số lượng khách hàng Exited và Non Exited của mỗi quốc gia')
ax[1].set_ylabel('count');
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090996403251187752/image.png?width=752&height=434">
</p>

```python
pd.crosstab(data.Geography,data.Exited,margins=True).style.background_gradient(cmap='gray_r')
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090996616967749672/image.png">
</p>
=>	The results show that the number of customers coming from France is the highest (50,14%). Next is Spain at 25.09% and finally is Germany is 24.77% with the highest customer leaving rate (Exited = 1) is 8.14%.
•	Statistics by Gender

```python
print(data.Gender.value_counts())
f,ax=plt.subplots(1,2,figsize=(18,8))
data['Gender'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Số lượng khách hàng theo giới tính')
ax[0].set_ylabel('count')
sns.countplot(data=data,x='Gender',hue='Exited',ax=ax[1])
ax[1].set_title('Số lượng khách hàng Exited và Non Exited theo mỗi giới tính')
ax[1].set_ylabel('count');
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090997005192540281/image.png?width=788&height=434">
</p>

=> The results show that the number of the male and female gender is quite equal. The ratio of leaving by Females is greater than the Ratio of leaving by Males.

• Statistics by Age

```python
Non_Exited = data[data['Exited']==0]
Exited = data[data['Exited']==1]

plt.subplots(figsize=(18,8))
sns.distplot(Non_Exited['Age'])
sns.distplot(Exited['Age'])
plt.title('Age:Exited vs Non Exited')
plt.legend([0,1],title='Exited')
plt.ylabel('percentage');
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090997335879864360/image.png?width=930&height=434">
</p>
=>	The age group over 40 is likely to leave the bank.

• Statistics by NumOfProducts

```python
pd.crosstab(data.NumOfProducts,data.Exited,margins=True).style.background_gradient(cmap='OrRd')
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090997557037113464/image.png">
</p>

```python
f,ax = plt.subplots(1,2,figsize=(18,8))
data['NumOfProducts'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number of customer by Number of Product')
ax[0].set_ylabel('count')
sns.countplot(data=data,x='NumOfProducts',hue='Exited',ax=ax[1])
ax[1].set_title('Number of Product:Exited vs Non Exited')
ax[1].set_ylabel('count');
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090997755515785216/image.png?width=935&height=434">
</p>

=> Statistics show that "1" = 5084 - accounting for 50.84%, "2" = 4590 accounting for 45.9%, "3" = 266 - accounting for 2.66% and "4" = 60 - accounting for 0.6%.

• Statistics by Credit score

```python
plt.figure(figsize=(18,8))
plt.hist(x='CreditScore',bins=100,data=Non_Exited,edgecolor='black',color='red')
plt.hist(x='CreditScore',bins=100,data=Exited,edgecolor='black',color='blue')
plt.title('Credit score: Exited vs Non-Exited')
plt.legend([0,1],title='Exited');
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090998031685529690/image.png?width=942&height=434">
</p>
=>	Customers who have a high credit score, tend to stay rather than leave.

### 2. Data Cleaning

- Because the attribute columns (RowNumber, CustomerId, Surname) are not necessary for data analysis, so can drop them.

```python
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
```

- Re-implement the similarity survey using the Pearson formula, combined with the Seaborn library to represent the graphical representation of the Heatmap graph results.

```python
plt.title("features correlation matrics".title(),
          fontsize=20,weight="bold")

sns.heatmap(data.corr(),annot=True,cmap='PuBu',linewidths=0.2, vmin=-1, vmax=1,linecolor = 'black')
fig=plt.gcf()
fig.set_size_inches(10,8);
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090998581609107557/image.png?width=493&height=434">
</p>

- Separate the prediction part (Exited) and dependent variables.

```python
features = data.drop('Exited', axis = 1)
labels = data['Exited']
```

- Check attribute columns with data types other than Number

```python
features.select_dtypes(exclude = ['number']).columns
```

- Convert data fields other than Number to a new value column with data type Number

```python
features_onehot = pd.get_dummies(features, columns = data.select_dtypes(exclude=['number']).columns)
features_onehot
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090998983511515186/image.png?width=1025&height=292">
</p>

- Check that the exited attribute has a data imbalance between 0 (7963 values) and 1(2037 values). This affects the predictive power of the model. Apply SMOTE method from imblearn.over_sampling library to handle imbalanced datasets.

```python
from imblearn.over_sampling import SMOTE
oversample = SMOTE(k_neighbors=4)
x_res, y_res = oversample.fit_resample(features_onehot, labels)
```

- Dataset has been balanced after using SMOTE method
<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1090999659754963014/image.png">
</p>

## III. Decision Tree

- Survey the similarity between columns with each other using Pearson's formula. Combine with seaborn graphing library to visualize data using heatmap graphs.
<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091000439543169145/image.png?width=658&height=434">
</p>
=> In this graph, the darker the color, the lower the similarity. Notice, no attributes are highly similar to each other, so it is not necessary to remove any columns.
- Then proceed to separate the data streams into two parts training and testing. Use the dataset after preprocessing of the data has been performed. Proceed to split the data into two parts train and test: (with training set will be 80%, and the testing set will be 20%).

```python
X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.20, random_state=0)
print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091000964980416592/image.png">
</p>

```python
print(y_train.value_counts(),'\n',y_test.value_counts())
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091001212184318082/image.png">
</p>
-	Build an ID3 tree based on the training data and test the results of the tree using the confusion matrix. Tree representation and in execution results. To build the ID3 tree execute the following statement:

```python
clf=tree.DecisionTreeClassifier(criterion="entropy",random_state=0)
clf.fit(X_train,y_train)
```

- With variablecriterion ="entropy" to ask the library to perform branching according to information gain.

<br>
- Obtained parameters of the ID3 tree:
  - Accuracy: 85%.
  -	Sensitivity: 85%.
  -	Coverage: 84%.

```python
tree_pred=clf.predict(X_test)
tree_score=metrics.accuracy_score(y_test,tree_pred)
print('Accuracy',tree_score)
print('Report:',metrics.classification_report(y_test,tree_pred))
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091001962545287249/image.png">
</p>

- Then we calculate the confusion matrix and represent it on the heatmap graph.

```python
tree_cm=metrics.confusion_matrix(y_test,tree_pred)

plt.figure(figsize=(12,12))
sns.heatmap(tree_cm,annot=True,fmt=".3f",linewidths=.5,square=True,cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
title='Decision Tree Accuracy Score: {0}'.format(tree_score)
plt.title(title,size=15)
```

```
Text(0.5, 1.0, 'Decision Tree Accuracy Score: 0.8487131198995606')
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091002380604153977/image.png?width=490&height=434">
</p>
• The graph tells us that the model predicts “yes” 1676 (281 + 1395) times and “no” 1510 (1309 + 201) times. While in fact 1590 (1309 + 281) times the mentioned event happened “yes” but predicted “no” and 1596 (201 + 1395) times happened “no” but predicted “yes”.
<br>
-	Represent the tree ID3 using code:

```python
fig,ax=plt.subplots(figsize=(50,24))
tree.plot_tree(clf,filled=True,fontsize=10)
plt.savefig('desision_tree',dpi=100)
plt.show()
```

<p align="center">
<img src="https://cdn.discordapp.com/attachments/847349555703316512/1091002660053856347/output.png">
</p>

### IV. Random Forest

- Import necessary libraries:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
```

- Using the Random Forest algorithm supported by the library.
- Configure predictive parameters for the model.

```python

//Số lượng cây
n_estimators = [int(x) for x in np.linspace(start = 5, stop = 800, num = 30)]

//Các tính năng được sử dụng để tách
max_features = ['auto', 'sqrt']

//Độ sâu của cây
max_depth = [int(x) for x in np.linspace(5, 2000, num = 30)]
max_depth.append(None)

//Lượng tối thiểu được đặt trong một nút trước khi nút được tách
min_samples_split = [2, 4, 6, 10]

//Lượng mẫu tối thiểu cần thiết để có ở một nút là
min_samples_leaf = [1,2, 4, 6, 10]

//Phương pháp lấy mẫu điểm dữ liệu
bootstrap = [True, False]

//Yêu cầu thư viện phân nhánh theo gini hoặc entropy
criterion = ['gini', 'entropy']
```

```python
rf = RandomForestClassifier()
rf_param = {
            "n_estimators" : n_estimators,
            "max_features" : max_features,
            "max_depth": max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap,
            "criterion": criterion
            }

cv_rf = StratifiedKFold(n_splits = 5) //Xác thực chéo

randomsearch_rf = RandomizedSearchCV(rf, rf_param, cv=cv_rf, scoring = "accuracy", n_jobs = -1, verbose = 2, n_iter = 10) //n_jobs: lượng công việc
                                                                                                                         //verbose: in thông tin khi đào tạo
randomsearch_rf.fit(X_train, y_train)

print("Độ chính xác cao nhất: ", randomsearch_rf.best_score_)
print("Lựa chọn các tham số tốt nhât:" , randomsearch_rf.best_params_)
```

```
  •	n_estimators: Number of trees the algorithm builds before averaging the predictions.
  •	max_features: Maximum number of features random forest considers splitting a node.
  •	max_depth: Maximum depth for each tree.
  •	min_samples_split: Minimum number of samples
  •	min_samples_leaf: Determines the minimum number of leaves required to split an internal node.
  •	criterion: Parameters to tell the library to the branch by entropy or gini.
```

- After running the algorithm, the output is as follows

```python
Fitting 5 folds for each of 10 candidates, totalling 50 fits
Độ chính xác cao nhất:  0.8939112674441191
Lựa chọn các tham số tốt nhât: {'n_estimators': 169, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 830, 'criterion': 'gini', 'bootstrap': True}
```
```
  •	best_score_ function will calculate the highest accuracy
  •	best_params_ function will choose the parameters to give the best results
    o	n_estimators: 169
    o	max_features: auto
    o	max_depth: 830
    o	min_samples_split: 2
    o	min_samples_leaf: 1
    o	criterion: Gini
    o	boostrap: True
```
-	Build models and check accuracy using actual and predicted values.
```python
print("Độ chính xác cao nhất: ", randomsearch_rf.best_score_)
print("Lựa chọn các tham số tốt nhât:" , randomsearch_rf.best_params_)
```
```python
Độ chính xác cao nhất:  0.8939112674441191
Lựa chọn các tham số tốt nhât: {'n_estimators': 169, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 830, 'criterion': 'gini', 'bootstrap': True}
```
___
```python
y_predicted_rf = randomsearch_rf.predict(X_test)

acc = accuracy_score(y_test, y_predicted_rf)
rp = classification_report(y_test, y_predicted_rf)

print("Độ chính xác của mô hình: " ,acc)
print(rp)
```

<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091004317210787972/image.png">
</p>

-	From the results, the model’s accuracy is 89.61%.\
-	In addition, there are other parameters to evaluate the model such as:\
    -	**Precision**: The ratio of the number of positive points the model predicts correctly to the total number of points the model predicts is Positive at 90%.
    -	**Recall**: The ratio of the number of Positive points that the model predicts correctly to the total number of points that are actually Positive at 90%.
    -	**F1-Score** is at 90%. Is the weighted average of Precision and Recall.<br>

- Then calculate the confusion matrix and display it on the Heatmap graph

```python
tree_score_rf=metrics.accuracy_score(y_test,y_predicted_rf)

plt.figure(figsize=(12,12))
sns.heatmap(confusion_matrix(y_test, y_predicted_rf),annot=True,fmt=".3f",linewidths=.5,square=True,cmap='YlGn_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
title='Random Forest Accuracy Score: {0}'.format(tree_score_rf)
plt.title(title,size=15)
```
```
Text(0.5, 1.0, 'Random Forest Accuracy Score: 0.8961079723791588')
```
<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091005507302588426/image.png?width=489&height=434">
</p>

-	The graph makes a total of 3183 predictions, including:
    - The model predicts “Yes” (Predicted = 1) is 1569 times, predicts “No” (Predict = 0) is 1617 times
    - In actuality, “Yes” (Actual = 1) is 1596 times, and “No” is 1590 times.

-	Use the predict function to predict the decision attribute of the random forest model
```python
y_predicted = randomsearch_rf.predict(features_onehot)
y_predicted
```
```
array([1, 0, 1, ..., 1, 0, 0], dtype=int64)
```
-	Merge the decision attribute column into the data
```python
prediction = pd.DataFrame(y_predicted, columns = ["Predicted"])
prediction_dataset = pd.concat([features_onehot,prediction], axis = 1)
prediction_dataset
```
<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091005942713286738/image.png?width=1025&height=294">
</p>

-	Compare the results between the actual and the predicted model.
```python
actual_data=data.iloc[:,[0, 10]]
actual_data.head(8)
```
<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091006161349775461/image.png">
</p>

```python
predict_data=prediction_dataset.iloc[:,[ 13]]
predict_data.head(8)
```
<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091006448814800936/image.png">
</p>

```python
pred_df = actual_data.join(predict_data.set_index(actual_data.index))

pred_df
```
<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091006476497190912/image.png">
</p>

```python
pred_df['prediction_accuracy'] = pred_df.apply(lambda x: "TRUE" if int(x['Exited'] == x['Predicted']) else "FALSE", axis=1)

pred_df['prediction_accuracy'].value_counts()
```
<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091006658676793364/image.png">
</p>

```python
_ = pd.DataFrame(pred_df, columns= ["prediction_accuracy"])
_ = pred_df.groupby('prediction_accuracy').size().plot(kind='bar', legend=True)
```
<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091006723659149375/image.png">
</p>
=> The prediction model gives 9698 true predictions and 302 false predictions.

## V. Result
### 1. Result statement
<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091007370886385744/image.png">
</p> <p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091007416512036865/image.png?width=532&height=434">
</p>
-	Compare confusion matrix results between algorithms
<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091007727427403967/image.png">
</p>
-	Compare model evaluation metrics
<p align="center">
<img src="https://media.discordapp.net/attachments/847349555703316512/1091007727767126116/image.png">
</p>
=> Random Forest model is better than Decision Tree model.

## VI. Compare algorithms and evaluate
-	Random Forest algorithm gives results with higher accuracy than the Decision Tree algorithm.
-	In addition, the Random Forest algorithm shows higher sensitivity as well as recall when processing on a larger data set than the Decision Tree algorithm.
-	The Decision Tree algorithm in the processing gives faster results and is easy to implement.
-	The Random Forest algorithm in the processing process gives more accurate and stable results.
