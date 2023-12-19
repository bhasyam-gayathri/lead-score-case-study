
# Lead score case study
An education company named X Education sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website and browse for courses

The company markets its courses on several websites and search engines like Google. Once these people land on the website, they might browse the courses or fill up a form for the course or watch some videos. When these people fill up a form providing their email address or phone number, they are classified to be a lead.
Now, although X Education gets a lot of leads, its lead conversion rate is very poor. For example, if, say, they acquire 100 leads in a day, only about 30 of them are converted. To make this process more efficient, the company wishes to identify the most potential leads, also known as ‘Hot Leads’.

The company requires you to build a model wherein you need to assign a lead score to each of the leads such that the customers with a higher lead score have a higher conversion chance and the customers with a lower lead score have a lower conversion chance.
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
warnings.simplefilter(action='ignore')
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\bhasyam.gayathri\Downloads\Lead+Scoring+Case+Study\Lead Scoring Assignment\Leads.csv")
data.head()
data.info()
data.isnull().sum().sort_values(ascending = False)
data.describe()
data['Converted'].value_counts()
# Check for categorical variable values
cat_cols = data.select_dtypes(exclude=['int64','float64']).columns
cat_cols = cat_cols.drop('Prospect ID')
for i in cat_cols:
    print(i)
    print(data[i].value_counts())
    print("***********************")
## Data Processing
#### Missing Value Treatment
Dropping the values with missing values >3000
data.drop(['Lead Quality','Asymmetrique Profile Score','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Activity Index','Tags'],axis = 1, inplace = True)
data.shape
data.isnull().sum().sort_values(ascending = False)
data['What is your current occupation'].fillna('Missing', inplace = True)
data['Specialization'].replace(['Select'],np.NaN, inplace=True)
data['Specialization'].fillna('Missing', inplace = True)
data['Specialization'].value_counts()
Removing the columns with 0 variance. i.e with only one value as they are of not much use in the prediction process.
# Removing the columns with one value i.e 0 variance

data.drop(['Do Not Call','What matters most to you in choosing a course','Magazine','Newspaper Article','X Education Forums','Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque','Country','City','Search'],axis = 1, inplace = True)
data.shape
We can also observe that some columns like 'Lead profile' and 'How did you hear about X Education' have a lost of values under select value. This columns also are of little use as this value is equal to NULL values. Hence dropping these two columns as well.
data.drop(['How did you hear about X Education','Lead Profile'], axis = 1, inplace = True)
data.shape
Dropping the columns Prospect ID and Lead number as they are unique values and cannot be useful for prediction
data.drop(['Prospect ID','Lead Number'],axis = 1, inplace = True)
data.shape
#Auto EDA
import sweetviz as sv
sweet_report = sv.analyze(data)
sweet_report.show_html('sweet_report.html')
#Impute missing values
#For Lead source and Last activity, We will be imputing with the mode value
#For Page Views Per Visit  and TotalVisits we will be imputing with the mean value

data['Lead Source'].fillna(data['Lead Source'].mode()[0],inplace=True)
data['Last Activity'].fillna(data['Last Activity'].mode()[0],inplace=True)
data['Page Views Per Visit'].fillna(data['Page Views Per Visit'].mean(),inplace=True)
data['TotalVisits'].fillna(data['TotalVisits'].mean(),inplace=True)
data.isnull().sum().sort_values(ascending = False)
## Data Visualisation
data_conv = data[data['Converted']==1]
data_not_conv = data[data['Converted']==0]
data_conv.info()
sns.distplot(data_conv['Total Time Spent on Website'], label = "Converted", hist = False)
sns.distplot(data_not_conv['Total Time Spent on Website'], label = "Not Converted", hist = False)
plt.legend()
plt.show()
Surprisingly people who are spending lot of time on the website are having high chances of being not converted and people who are spending moderately on the website are having high chances of being converted
plt.figure(figsize = (30,7))
plt.subplot(2,1,1)
plt.title("Converted")
sns.countplot(x = "What is your current occupation", data = data_conv, order = data_conv['What is your current occupation'].value_counts().index);

plt.subplot(2,1,2)
plt.title("Not Converted")
sns.countplot(x = "What is your current occupation", data = data_not_conv, order = data_not_conv['What is your current occupation'].value_counts().index);
We can see that working professional have the high chances of being converted after unemployed
plt.figure(figsize = (30,7))
plt.subplot(2,1,1)
plt.title("Converted")
sns.countplot(x = "Lead Origin", data = data_conv, order = data_conv['Lead Origin'].value_counts().index);

plt.subplot(2,1,2)
plt.title("Not Converted")
sns.countplot(x = "Lead Origin", data = data_not_conv, order = data_not_conv['Lead Origin'].value_counts().index);
### Dummy Variable Creation
Now we will deal the categorical columns by creating dummy variables
cat_col = data.select_dtypes(include= 'object').columns
cat_col
dummy = pd.get_dummies(data[['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
       'Specialization', 'What is your current occupation',
       'A free copy of Mastering The Interview', 'Last Notable Activity']],drop_first= True)
data = pd.concat([dummy,data],axis = 1)
data.drop(cat_col,axis = 1, inplace = True)
data.head()
#Separating the target variable
X = data.drop(['Converted'],axis =1)
y = data['Converted']
#Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)
# Scaling

Lets scale the numeric columns using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
#Lets scale the three numeric features present

scaler = MinMaxScaler()
X_train[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']])
X_train.head()
#Checking for correlation between columns
X_corr = X.corr().abs().unstack()
X_corr.sort_values(ascending=True, inplace=True)
columns_above_80 = [(col1, col2) for col1, col2 in X_corr.index if X_corr[col1,col2] > 0.8 and col1 != col2]
print(columns_above_80)
#X_corr.loc[np.where(X.corr()>0.8, 1, 0)==1].columns
X_corr.sort_values(ascending=False)
#Auto EDA
import sweetviz as sv
sweet_report = sv.analyze(data)
sweet_report.show_html('sweet_report.html')
#Removing the columns with >0.8 correlation
#X_train.drop(['Last Notable Activity_Email Link Clicked', 'Last Activity_Email Link Clicked','Last Activity_Email Opened', 'Last Notable Activity_Email Opened', 'Last Notable Activity_SMS Sent', 'Last Activity_SMS Sent','Lead Origin_Lead Add Form', 'Lead Source_Reference', 'Last Notable Activity_Unsubscribed', 'Last Activity_Unsubscribed','Lead Source_Facebook', 'Lead Origin_Lead Import', 'Last Activity_Email Marked Spam', 'Last Notable Activity_Email Marked Spam', 'Last Activity_Resubscribed to emails', 'Last Notable Activity_Resubscribed to emails'], axis = 1, inplace = True)
#X_train.shape
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import model_selection, tree, ensemble, metrics
lr = LogisticRegression()
lr.fit(X_train,y_train)
print("Train score :")
print(lr.score(X_train,y_train))
col = X_train.columns
X_test[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']])
print("Test score :")
print(lr.score(X_test[col],y_test))
pred = lr.predict(X_test[col])

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
X_test[col].shape
lr.predict_proba(X_test[col])
from sklearn.feature_selection import RFE
rfe = RFE(lr,15)
rfe = rfe.fit(X_train,y_train)
rfe
#Lets look at the feature selected by RFE
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train = X_train[col]
X_train.shape
import statsmodels.api as sm


X_train_sm = sm.add_constant(X_train)
lr2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = lr2.fit()
res.summary()
There are some columns with P values greater than 0.05 but before removing them lets look at VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.astype(float).values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
All the variables have VIF below 5. Hence dropping only the column "What is your current occupation_Housewife" which has high p value
X_train.drop(['What is your current occupation_Housewife'],axis = 1, inplace = True)
lr3 = sm.GLM(y_train,X_train, family = sm.families.Binomial())
res = lr3.fit()
res.summary()
Drop the column 'Last Activity_Had a Phone Conversation' with high p value and check the model again.
X_train.drop(['Last Activity_Had a Phone Conversation'],axis = 1, inplace = True)
lr4 = sm.GLM(y_train,X_train, family = sm.families.Binomial())
res = lr4.fit()
res.summary()
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.astype(float).values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
Drop the column 'TotalVisits' with high P values and check the model again
X_train.drop(['TotalVisits'],axis = 1, inplace = True)
lr5 = sm.GLM(y_train,X_train, family = sm.families.Binomial())
res = lr5.fit()
res.summary()
Now both the p values and VIF are below the limit. Now we can proceed with the model evaluation
y_train_pred = res.predict(X_train)
y_train_pred
First choosing the cut off as 0.5 and labeling them as converted/not converted and then decide the cut off after checking the model metrics for different cust offs
y_train_pred_final = pd.DataFrame({'Converted': y_train.values, 'Conversion_prob':y_train_pred})
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_prob.map(lambda x: 1 if x>0.5 else 0)
#y_train_pred_final['Prob'] = round(y_train_pred_final.Conversion_prob,1)

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    y_train_pred_final[i] = y_train_pred_final.Conversion_prob.map(lambda x: 1 if x>i else 0)

y_train_pred_final
#Let evaluate the model with different metrics
from sklearn import metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Converted,y_train_pred_final.Predicted)
confusion
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
#Sensitivity
Sensitivity = TP/(TP+FN)
#Specificity
Specificity = TN/(TN+FP)
print ("Sensitivity :"+str(Sensitivity))
print("Specificity : "+str(Specificity))
# Finding optimal cutoff
We choose 0.5 value as cut off lets see what is the correct cut off we can go for 
from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve(y_train_pred_final.Converted,y_train_pred_final.Conversion_prob)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
The area under the curve is 0.88 which is pretty decent
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
cutoff_df = pd.DataFrame(columns = ['prob','accuracy','sensi','speci'])
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i])
    total = sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,1]+cm1[1,0])
    cutoff_df.loc[i] = [i,accuracy,sensi,speci]
print(cutoff_df)
cutoff_df.plot.line(x='prob', y = ['accuracy','sensi','speci'])
At the point 0.38 we can see all the metrics are getting merged. So we can choose 0.38 as the cut off
y_train_pred_final['final_Predicted'] = y_train_pred_final.Conversion_prob.map(lambda x: 1 if x>0.38 else 0)
y_train_pred_final
metrics.accuracy_score(y_train_pred_final.Converted,y_train_pred_final.final_Predicted)
## Making Predictions on Test data
#Final test predictions
col = X_train.columns
X_test = X_test[col]
lrf = LogisticRegression()
lrf.fit(X_train,y_train)
print("Train score :")
print(lrf.score(X_train,y_train))
print("Test score :")
print(lrf.score(X_test,y_test))
pred = lrf.predict(X_test)
pred
pred_prob = lrf.predict_proba(X_test)
pred_prob[0:,0]
y_test_pred = pd.DataFrame()
y_test_pred['Converted'] = pred
y_test_pred['Converted_prob'] = pred_prob[0:,0]
y_test_pred
y_test
confusion = metrics.confusion_matrix(y_test_pred.Converted,y_test)
confusion
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
#Sensitivity
Sensitivity = TP/(TP+FN)
#Specificity
Specificity = TN/(TN+FP)
#Accuracy
Accuracy = (TP+TN)/(TP+TN+FP+FN)
print ("Sensitivity :"+str(Sensitivity))
print("Specificity : "+str(Specificity))
print("Accuracy : "+str(Accuracy))
