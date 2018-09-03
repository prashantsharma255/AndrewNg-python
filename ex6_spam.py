# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:08:15 2018

@author: Hp
"""
from numpy import loadtxt
from scipy.io import loadmat
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()

file_contents =open(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex6\ex6\emailSample1.txt').read()

def processEmail(email_contents):
    #load vocabulary
    vocabList = getVocabList()
    word_indices= []
    tokenList=[]
    
    #preprocess email
    email_contents =  email_contents.lower()   #Lower casing
    email_contents = re.sub('<[^<>]+>','',email_contents)     #stripping html
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr',email_contents) #Normalizing urls
    email_contents = re.sub('[a-z0-9]+@[a-z]+\.com','emailaddr',email_contents) #normalizing email #[^\s]+@[^\s]+
    email_contents = re.sub('[0-9]+','number',email_contents)  #normalizing numbers
    email_contents = email_contents.replace('$','dollar')  #normalizing dollar
    email_contents = re.sub('[^a-z0-9]',' ',email_contents)  #removing non alphanumeric
    tokenList = word_tokenize(email_contents)   #tokenizing using nltk
    tokenList = [ps.stem(x) for x in tokenList]  # stemming
    
    for x in tokenList:   #eliminate if length is too short
        if(len(x)<1):
            continue
        
    word_indices = [vocabList.index(x) for x in tokenList if x in vocabList]   #returning index of words
    
    return word_indices
        
def getVocabList():
    fid = open(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex6\ex6\vocab.txt').readlines()
    words=[]
    
    for line in fid:
        a,b = line.split()
        words.append(b)
        
    return words

word_indices = processEmail(file_contents)
#print(word_indices)

def emailFeatures(word_indices):  #extracting features from email
    n=1899
    x = np.zeros(n)
    
    for idx in word_indices:
        x[idx]=1
        
    return x.reshape([n,1])

x= emailFeatures(word_indices)

#training data
data = loadmat(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex6\ex6\spamTrain.mat')
X = data['X']
y = data['y']
clf = svm.SVC(C=1,kernel='linear').fit(X,y.ravel())
p = clf.predict(X)
p = p.reshape([len(p),1])
acc = (p==y).mean()*100
print("Training accuracy is :")
print(acc)

#testing data
data2 = loadmat(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex6\ex6\spamTest.mat')
Xtest = data2['Xtest']
ytest = data2['ytest']
p2 = clf.predict(Xtest)
p2 = p2.reshape([len(p2),1])
acc2 = (p2==ytest).mean()*100
print("Test accuracy is :")
print(acc2)

#top predictors of spam
idx = np.argsort(clf.coef_)
vocabList = getVocabList()
idx = idx.reshape([len(idx[0]),1])
print([vocabList[int(x)] for x in idx[-15:]])

#try your own mails
sampleEmail = open(r'C:\Users\Hp\Desktop\Mlclass\machine-learning-ex6\ex6\spamSample1.txt').read()
#print(sampleEmail)
word_indices2 = processEmail(sampleEmail)
x2 = emailFeatures(word_indices2)
x2 = x2.reshape([1,-1])
p3 = clf.predict(x2)
print(p3)
