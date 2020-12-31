#!/usr/bin/env python
# coding: utf-8

# In[131]:


import pandas as pd
data=pd.read_csv('bank-full.csv',sep=';')
data.head()


# In[132]:


data.tail()


# In[133]:


data.info()


# In[134]:


data.shape


# In[135]:


data.columns


# In[136]:


data=data.dropna()
data.shape


# In[137]:


data['age'].unique()


# In[138]:


data['job'].unique()


# In[139]:


data['marital'].unique()


# In[140]:


data['education'].unique()


# In[141]:


data['default'].unique()


# In[142]:


data['balance'].unique()


# In[143]:


data['housing'].unique()


# In[144]:


data['loan'].unique()


# In[145]:


data['contact'].unique()


# In[146]:


data['day'].unique()


# In[147]:


data['month'].unique()


# In[148]:


data['duration'].unique()


# In[149]:


data['campaign'].unique()


# In[150]:


data['pdays'].unique()


# In[151]:


data['previous'].unique()


# In[152]:


data['poutcome'].unique()


# In[153]:


data['y'].unique()


# In[154]:


data['y'].value_counts()


# In[155]:


data['y']=data['y'].map({'yes':1,'no':0})


# In[156]:


data['y'].value_counts()


# In[157]:


import seaborn as sns
sns.countplot(x=data['y'],palette='hls')
plt.show()


# In[33]:


not_count=len(data[data['y']==0])
print (not_count)


# In[158]:


count=len(data[data['y']==1])
print (count)


# In[159]:


not_sub=not_count/(not_count+count)
print('Not confirm',not_sub*100)


# In[160]:


sub=count/(not_count+count)
print('Confirm',sub*100)


# In[161]:


data.groupby('y').mean()


# In[162]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(data.job,data.y).plot(kind='bar')
plt.xlabel('Job')
plt.ylabel('Frequency of purchase of term deposit')
plt.title('Purchase frequency for Job title')


# In[163]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(data.marital,data.y).plot(kind='bar')
plt.xlabel('Marital')
plt.ylabel('Frequency of purchase of term deposit')
plt.title('Purchase frequency for Marital title')


# In[164]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
table=pd.crosstab(data.education,data.y)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.xlabel('Education')
plt.ylabel('Frequency of purchase of term deposit')
plt.title('Purchase frequency for Education title')


# In[165]:


data['age'].hist()


# In[166]:


#create dummy variables
cat_vars=['job','marital','education','default','housing','loan','contact','month','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+'var'
    cat_list=pd.get_dummies(data[var],prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['job','marital','education','default','housing','loan','contact','month','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


# In[167]:


data_final=data[to_keep]
data_final.columns.values


# In[168]:


data_final


# In[54]:


get_ipython().system('pip install imblearn')


# In[169]:


#SMOTE
X=data_final.loc[:,data_final.columns != 'y']
Y=data_final.loc[:,data_final.columns == 'y']

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

os=SMOTE(random_state=0)
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3, random_state=0)
columns=X_train.columns

os_data_X, os_data_Y=os.fit_sample(X_train, Y_train)
os_data_X=pd.DataFrame(data=os_data_X,columns=columns)
os_data_Y=pd.DataFrame(data=os_data_Y,columns=['y'])

print("Length of oversampled data is", len(os_data_X))
print('No. of no sub in oversampled data',len(os_data_Y[os_data_Y['y']==0]))
print('No. of sub in oversampled data',len(os_data_Y[os_data_Y['y']==1]))
print('Proportion of no sub in oversampled data',len(os_data_Y[os_data_Y['y']==0])/len(os_data_X))
print('Proportion of sub in oversampled data',len(os_data_Y[os_data_Y['y']==1])/len(os_data_X))


# In[170]:


X.columns


# In[171]:


data_final_vars=data_final.columns.values.tolist()
Y=data['y']
X=[i for i in data_final_vars if i not in Y]

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()

rfe=RFE(logreg,20)
rfe=rfe.fit(os_data_X,os_data_Y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)


# In[172]:


cols=['job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management','job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'education_primary', 'education_secondary', 'education_tertiary',
       'education_unknown','housing_no','housing_yes']
Y=os_data_Y['y']
X=os_data_X[cols]


# In[173]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

logreg.fit(X_train,Y_train)

y_pred=logreg.predict(X_test)


# In[174]:


y_pred


# In[182]:


print('Accuracy on train set:',logreg.score(X_train,Y_train))


# In[181]:


print('Accuracy on test set:',logreg.score(X_test,Y_test))


# In[175]:


from sklearn import metrics
cnf=metrics.confusion_matrix(Y_test,y_pred)
cnf


# In[176]:


#Using Heatmap
import seaborn as sns
sns.heatmap(pd.DataFrame(cnf),annot=True, cmap='YlGnBu',fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')


# In[177]:


print('Accuracy:',metrics.accuracy_score(Y_test,y_pred))


# In[178]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))


# In[179]:


#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc=roc_auc_score(Y_test,logreg.predict(X_test))
fpr,tpr,thresholds=roc_curve(Y_test,logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label='Logistic Regression (area=%0.2f)'% logit_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0] )
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating characteristic')
plt.legend(loc='lower right')
plt.show()


# In[180]:


#Implementing the model
import statsmodels.api as sm
logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary2())


# In[ ]:




