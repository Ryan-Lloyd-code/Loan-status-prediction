# -*- coding: utf-8 -*-
"""
Created on Sun May 24 19:16:08 2020

@author: gilli
"""
#%%
import pandas as pd


#read in description of data
data_info = pd.read_csv('DATA/lending_club_info.csv',index_col='LoanStatNew')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read in data
df = pd.read_csv('DATA/lending_club_loan_two.csv')

#See parameters of data
information = df.info()
#%%

#see balance of the output
plt.figure(figsize=(12,5))
sns.countplot(x='loan_status',data=df)

#see distribution of loan amounts
plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False,bins=40)
plt.xlim(0,45000)

#visualise correaltions between parameters
plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.ylim(10, 0)

plt.figure(figsize=(12,7))


sns.scatterplot(x='installment',y='loan_amnt',data=df,)

#see if there's a relation between amount of loan and statuts ... no real relation
sns.boxplot(x='loan_status',y='loan_amnt',data=df)
#numerical summary of above diagram
print(df.groupby('loan_status')['loan_amnt'].describe())

#Look at the Grade and sub grade categories

print(sorted(df['grade'].unique()))
print(sorted(df['sub_grade'].unique()))

plt.figure(figsize=(12,7))
sns.countplot(x = 'grade',  data=df, hue='loan_status')

plt.figure(figsize=(12,7))
sns.countplot(x = 'sub_grade',  data=df[(df['grade']=='F') | (df['grade']=='G')], hue='loan_status')
plt.figure(figsize=(12,7))
subgrade_order = sorted(df[(df['grade']=='F') | (df['grade']=='G')]['sub_grade'].unique())
sns.countplot(x = 'sub_grade',  data=df[(df['grade']=='F') | (df['grade']=='G')], hue='loan_status', order=subgrade_order)

#change loan status to boolean values

print(df['loan_status'].unique())

df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
loan_repaid_corr=df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind = 'bar')
#%%
#MISSING DATA

print(df.isnull().sum())

#too many occupations 
print(df['emp_title'].nunique())
print(df['emp_title'].value_counts())

df = df.drop(['emp_title'], axis = 1)


print(sorted(df['emp_length'].dropna().unique()))
emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']


plt.figure(figsize=(12,7))
sns.countplot(x = 'emp_length',data=df, hue = 'loan_repaid', order=emp_length_order)

emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_co/emp_fp
print(emp_len)
# no real significance in emplyment length.. therefore drop column
df = df.drop('emp_length',axis=1)

print("Correlation with the mort_acc column")
print(df.corr()['mort_acc'].sort_values())

total_acc_avg=df.groupby('total_acc').mean()['mort_acc']
def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc
    
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']),axis = 1 )

df = df.dropna()

#feature engineering categories

df['term'] = df['term'].apply(lambda term: int(term[:3])) # get montsh as integer

df = df.drop('grade',axis=1) # already have subgrade

#get dummies of subgrade, drop first subgrade

subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)

# see what objects exist and which objects can become dummies
df.select_dtypes(['object']).columns

dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

df['home_ownership'].value_counts()
df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

#get only the zip code
df['zip_code'] = df['address'].apply(lambda address:address[-5:])
dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)

df = df.drop('issue_d',axis=1)

#Get only the year
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)
df = df.drop('loan_status',axis=1)
df = df.drop('title',axis=1)
#%%
#create training and test data

from sklearn.model_selection import train_test_split

X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values

# df = df.sample(frac=0.1,random_state=101)
#print(len(df))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.preprocessing import MinMaxScaler

#scale the data

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##TENSORFLOW

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

model = Sequential()
# input layer
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='sigmoid'))

# Compile model 
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          )


#save model
from tensorflow.keras.models import load_model
model.save('full_data_project_model.h5')  

#plot performance per iteration

losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()

from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))