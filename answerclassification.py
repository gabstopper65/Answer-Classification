#!/usr/bin/env python
# coding: utf-8

# 
# BT4222 Kaggle Competition Code
# 

#import all the necessary packages here
import pandas as pd
import numpy as np
import nltk

path = '/content/drive/My Drive/BT4222/train.csv'
train = pd.read_csv(path)
#always a good idea to inspect the data
train.head()
train.shape

path = '/content/drive/My Drive/BT4222/test.csv'
test = pd.read_csv(path)
#always a good idea to inspect the data
test.head()
test.shape

# Step 1 : Data Preprocessing
#check for na values
train.isnull().sum()

#Check for balanced classes
train_outcome = train.groupby('Outcome',as_index=False).count()
train_outcome


# This is where text preprocessing is done. Eg Lemmatization, removal of stopwords, lowercase

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

#I create a class for the lemmatizer to be used as a tokenizer.
class LemmaTokenizer(object):
    def __init__(self):
        self.word = WordNetLemmatizer()
    def __call__(self, documents):
        return [self.word.lemmatize(x) for x in word_tokenize(documents)]

# Step 2 : Feature Engineering. Many features were created using the tfidvectoriser.
from sklearn.feature_extraction.text import TfidfVectorizer
vect_word = TfidfVectorizer(tokenizer = LemmaTokenizer(),max_features=20000, lowercase=True, analyzer='word',
                        stop_words= 'english',ngram_range=(1,3),dtype=np.float32,max_df = 0.9)

import nltk
nltk.download('punkt')
nltk.download('wordnet')
tr_vect = vect_word.fit_transform(train['Comment'].values.astype('U'))
ts_vect = vect_word.transform(test['Comment'].values.astype('U'))
X = tr_vect

# Splitting of dataset into train and validation set
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, train["Outcome"], random_state=1)


# Different models was attempted on the dataset. Model 1 : Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1,random_state = 1,class_weight = 'balanced',max_iter=1000)
lr.fit(X_train,y_train)

# Evaluate the model on the validation set. Test using accuracy_score and roc_score
pred_Y_valid = lr.predict(X_valid)
from sklearn import metrics
print(metrics.accuracy_score(y_valid, pred_Y_valid))
print(metrics.roc_auc_score(y_valid, pred_Y_valid))


# Model 2 : Support Vector Machines
from sklearn import svm
SVM = svm.SVC(kernel="linear",random_state=1)
SVM.fit(X_train,y_train)

# Evaluate the model on the validation set. Test using accuracy_score and roc_score
pred_Y_valid = SVM.predict(X_valid)
from sklearn import metrics
print(metrics.accuracy_score(y_valid, pred_Y_valid))
print(metrics.roc_auc_score(y_valid, pred_Y_valid))


# Model 3 : Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000,random_state = 1,min_samples_split = 5)
rf.fit(X_train, y_train)


# Evaluate the model on the validation set. Test using accuracy_score and roc_score
pred_Y_valid = rf.predict(X_valid)
from sklearn import metrics
print(metrics.accuracy_score(y_valid, pred_Y_valid))
print(metrics.roc_auc_score(y_valid, pred_Y_valid))


# Model 4 : Voting Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
lr = LogisticRegression(C=1,random_state = 1,class_weight = 'balanced',max_iter=1000)
rf = RandomForestClassifier(n_estimators=1000, random_state=1,verbose =100)
SVM = svm.SVC(kernel="linear",random_state=1)
eclf = VotingClassifier(estimators=[('lr', lr), ('rf', rf),('svm',SVM)], voting='hard')
eclf.fit(X_train,y_train)


# Evaluate the model on the validation set. Test using accuracy_score and roc_score
pred_Y_valid = eclf.predict(X_valid)
from sklearn import metrics
metrics.accuracy_score(y_valid, pred_Y_valid)
metrics.roc_auc_score(y_valid, pred_Y_valid)


# Final chosen model 5 : Stacking Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn import svm
estimators = [('rf', RandomForestClassifier(n_estimators=1000,verbose = 100, random_state=1)),('svr', svm.SVC(kernel="linear",random_state=1))]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(C=1,random_state = 1,class_weight = 'balanced',max_iter=1000))
clf.fit(X_train,y_train)


# Evaluate the model on the validation set. Test using accuracy_score and roc_score
pred_Y_valid = clf.predict(X_valid)
from sklearn import metrics
metrics.accuracy_score(y_valid, pred_Y_valid)
metrics.roc_auc_score(y_valid, pred_Y_valid)


# Last step: 
# Applying the model on my test set
# Note that ts_vect is the features from the test set
# from above
#ts_vect = vect_word.transform(test['Comment'].values.astype('U'))
pred_Y_test = clf.predict(ts_vect)
pred_Y_test


# Generating my submission
submission = pd.DataFrame({'Id': test['Id'].values, 'Outcome': pred_Y_test})
submission.to_csv('/content/drive/My Drive/BT4222/A0172579N_submission19.csv', index=False)

