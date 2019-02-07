
# coding: utf-8

# In[1]:


# All the import statements

import os
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
import urllib.request
from nltk.corpus import stopwords
import re
import tarfile
from string import punctuation
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[2]:


# Declaring various classes of the newsgroups

a=[] ## list decleration for file names

classes=["alt.atheism","comp.graphics","comp.os.ms-windows.misc","comp.sys.ibm.pc.hardware",
         "comp.sys.mac.hardware","comp.windows.x","misc.forsale","rec.autos","rec.motorcycles",
         "rec.sport.baseball","rec.sport.hockey","sci.crypt","sci.electronics","sci.med",
         "sci.space","soc.religion.christian","talk.politics.guns","talk.politics.mideast",
         "talk.politics.misc","talk.religion.misc"]

print(len(classes))


# In[3]:


## Extracting the stopwords from the nltk library
nltk.download('stopwords')
StopWords=set(stopwords.words('english'))
print(StopWords)

# Our own list of some block words to be avoided; observed from the documents

block_words = ['newsgroups', 'xref', 'path', 'from', 'subject', 'sender', 'organisation', 'apr',
               'gmt', 'last','better','never','every','even','two','good','used','first','need',
               'going','must','really','might','well','without','made','give','look','try','far',
               'less','seem','new','make','many','way','since','using','take','help','thanks','send',
               'free','may','see','much','want','find','would','one','like','get','use','also','could',
               'say','us','go','please','said','set','got','sure','come','lot','seems','able','anything',
               'put', '--', '|>', '>>', '93', 'xref', 'cantaloupe.srv.cs.cmu.edu', '20', '16', 
               "max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'", '21', '19', '10', 
               '17', '24', 'reply-to:', 'thu', 'nntp-posting-host:', 're:','25''18'"i'd"'>i''22''fri,''23''>the',
               'references:','xref:','sender:','writes:','1993','organization:']


# In[5]:


# declaring a set of special characters and numbers
punc = (set(punctuation))
print (punc)
num = {'0','1','2','3','4','5','6','7','8','9'}


# In[14]:


data={}
data["train"]={}
data["test"]={}
for i in range(20):
    s=classes[i]
    data[s]=[]


# In[7]:


## statement to retrieve the dataset from the link

urllib.request.urlretrieve ("https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz", "a.tar.gz")


# In[8]:


## Statements to extract the database documents from web

tar = tarfile.open("a.tar.gz")
tar.extractall()
tar.close()


# In[15]:


## adding all the file names to a list

for i in range(20):
    a.clear()
    for files in os.listdir("./20_newsgroups/"+classes[i]):
        data[classes[i]].append(files)


# In[16]:


# temporary list declaration

word_present={}
alternate_word_array=[]


# In[17]:


## This piece of code will extract the words from the documents and calculate
## their occurences in a dictionary

for i in range(20): ## number of newsgroups
    for j in range(len(data[classes[i]])):
        
        ## declaring the path of every individual file
        path = "./20_newsgroups/"+classes[i]+"/"+data[classes[i]][j]
        
        ## opening the file
        text = open(path, 'r', errors='ignore').read()
        
        ## going through every word in the file
        for word in text.split():
            word=word.lower() ## converting the word to lowercase
            
            ## making sure word is not a stopword and not in the list of our block words
            if word not in StopWords and word not in block_words:
                
                ## claculating the frequency of the word
                if word in word_present and word:
                    word_present[word]+=1
                else:
                    word_present[word]=1
                    alternate_word_array.append(word)


# In[18]:


print(len(word_present.keys()))
print(type(alternate_word_array))
print(type(alternate_word_array[0]))


# In[13]:


## this piece of code was used to refine the words in our dictonary
## that is to remove the special characters present in some words so that we can
## work only with pure words.

## But using this code gives a lower accuracy so not using.
## seems like this ML model doesn't want to work with cleaner data :)

x=[]
for s in alternate_word_array:
    last=0
    word_array=[]
    j=0
    for i in range(len(s)):
        if s[i] in punc:
            j+=1
            if last!=i:
                word_array.append(s[last:i])
            last = i+1
    if last != len(s):
        word_array.append(s[last:])
    if len(word_array)>=2:
        for c in word_array:
            if c in word_present:
                word_present[c]+=1
            else:
                word_present[c]=1
    if j>0:
        x.append(s)
        
for i in x:
    del word_present[i]
    
print(len(word_present.keys()))


# In[19]:


## part of the previous cell, used this for cleaning.
## not using this so please ignore.

x.clear()
for s in word_present.keys():
    for i in range(len(s)):
        if s[i] in num:
            x.append(s)
            break
for i in x:
    del word_present[i]
print(len(word_present.keys()))


# In[19]:


## using only the words which occur more that 200 times in the 20000 documents

x.clear()
for s in word_present.keys():
    if word_present[s] <= 200:
        x.append(s)        

## deleting less frequency words from the dictionary

for i in x:
    del word_present[i]
print(len(word_present.keys()))


# In[20]:


## making a final list of the words we are using

final_words=[]
for i in word_present.keys():
    final_words.append(i)


# In[21]:


print(final_words)


# In[22]:


## time to convert our dictionary to numpy matrix

database = np.zeros((19997,len(final_words)))


# In[23]:


## this function was supposed to accompany the previous data cleaning efforts I made
## Like those functions too not using this code

def clean(word):
    while len(word)>0 and word[-1] in punc:
        word = word[:-1]
    while len(word)>0 and word[0] in punc:
        word = word[1:]
    return word


# In[24]:


## Similar story like the previous cell

def no_num(s):
    for i in range(len(s)):
        if s[i] in num:
            return 1
    return 0


# In[25]:


## this piece of code will again read the words fro every document and help us to
## convert our dictionary into a numpy matrix which can be used for multinomialNB

counter = 0
for i in range(20):
    for j in range(len(data[classes[i]])):
        
        ## declaring the path for every individual file
        path = "./20_newsgroups/"+classes[i]+"/"+data[classes[i]][j]
        
        ## opening the file
        text = open(path, 'r', errors='ignore').read()
        
        ## going through every word in the file
        for word in text.split():
            word=word.lower()
            
            ## making sure word is not a stopword and not in the list of our block words
            if word not in StopWords and word not in block_words:
                if word in final_words:
                    
                    ## adding the word frequency to the matrix
                    idx = final_words.index(word)
                    database[counter][idx] += 1
        counter += 1
print(counter)


# In[26]:


sum_array = np.sum(database,axis=0)


# In[27]:


## this piece of code is used to assign the class to every datapoint in the 
## database the we created

y = []
for i in range(len(classes)):
    files = os.listdir('./20_newsgroups/' + classes[i])
    for j in range(len(files)):
        y.append(i)
y = np.array(y)
y.shape


# In[28]:


## test train splitting the database

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(database, y, test_size = 0.25, random_state = 0)


# In[29]:


## applying the multinomialNB from sklearn for our predictions

clf = MultinomialNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

## calculating the training and testing score

train_score = clf.score(x_train, y_train)
test_score = clf.score(x_test, y_test)

train_score, test_score


# In[30]:


f_list = final_words


# In[31]:


## the fit function for our own naive bayes

def fit(x_train, y_train):

    count={}
    set_class = set(y_train)            
    for current_class in set_class:
        count[current_class] = {}
        count["total_data"] = len(y_train)
        
        ##Rows whose class is current_class
        current_class_rows = (y_train == current_class)
        
        x_train_current = x_train[current_class_rows]
        y_train_current = y_train[current_class_rows]
        
        sums = 0
        for i in range(len(f_list)):
            ## For each class, calculating total frequency of a feature 
            count[current_class][f_list[i]] = x_train_current[:,i].sum()
            sums = sums + count[current_class][f_list[i]]
        
        ##Calculating total count of words of a class
        count[current_class]["total_count"] = sums
        
    return count


# In[32]:


def probability(dictionary, row, current_class):
    ## class_prob = log of probability of the current class = log(no of documents having class as current_class)/ (total number of documents)
    class_prob = np.log(dictionary[current_class]["total_count"]) - np.log(dictionary["total_data"])
    total_prob = class_prob
    
    
    for i in range(len(row)):
        ##Numerator
        word_count = dictionary[current_class][f_list[i]] + 1     
        ## Denominator
        total_count = dictionary[current_class]["total_count"] + len(f_list)
        ## Add 1 to numerator and len(row) in denominator for laplace correction
        
        ## Log Probabilty of a word 
        word_prob = np.log(word_count) - np.log(total_count)
        
        ##Calculating probability frequency number of times
        for j in range(int(row[i])):
            total_prob += word_prob
        
    return total_prob


# In[33]:


def predictSinglePoint(row, dictionary):
    classes = dictionary.keys()
    
    ##Initialising best_prob and best_class as very low count
    
    best_prob = -1000
    best_class = -1
    first_iter = True
    
    for current_class in classes:
        if(current_class == "total_data"):
            continue
        
        ##Calculating probabilty that the given row belong to current_class
        prob_current_class = probability(dictionary, row, current_class)
        
        ##For first iteration we set the best_prob to be the probabilty that row is of first class and best_class to be first class
        ##For rest iteration, we check if the probabilty that row is of the current_class is greater than the best_prob then we update best_prob and best_class.
        if(first_iter or prob_current_class > best_prob):
            best_prob = prob_current_class
            best_class = current_class
        
        first_iter = False
    
    ## Return the best class which has maximum probabilty.
    return best_class


# In[34]:


def predict(x_test, dictionary):
    ## Initialise a list which contain the predictions
    y_pred_self = []
    
    ##Iterate through each row in x_test
    for j in range(len(x_test)):
        
        ##Calculate the prediction of the class to which the row belong to.
        pred_class = predictSinglePoint(x_test[j,:], dictionary) 
        
        ##Append the predicted class to our list
        y_pred_self.append(pred_class)
    
    ##Return the list of predictions
    return y_pred_self


# In[35]:


dictionary = fit(x_train, y_train)

##Testing the model 
y_pred_self = predict(x_test, dictionary)


# In[36]:


## comparing the accuracy of the two models

print("Accuracy for self-implemented Naive Bayes - ", accuracy_score(y_test, y_pred_self))
print("Accuracy for sklearn MultinomialNB() - ", test_score)


# In[37]:


## comparing the classification report of the two algorithms

print("Classification report for sklearn MultinomialNB()",classification_report(y_test, y_pred))
print("Classification report for self-implemented Naive Bayes ",classification_report(y_test, y_pred_self))

