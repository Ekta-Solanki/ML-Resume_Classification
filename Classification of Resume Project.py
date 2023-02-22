#!/usr/bin/env python
# coding: utf-8

# Steps for resume classification
# 1. Import Library
# 2. Data Preprocessing
# 3. Performing EDA
# 4. Creating wordcloud using wordcloud library
# 5. Model building and displaying classification report

# In[77]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import re


# In[78]:


resume_data = pd.read_csv('https://github.com/Ekta-Solanki/ML-Resume_Classification/blob/681567bcff66329d81d5e18637085b71b7e43101/UpdatedResumeDataSet.csv')


# In[79]:


resume_data.head()


# # 2. Data Preprocessing

# In[80]:



def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText


# In[81]:


resume_data['cleaned'] = resume_data['Resume'].apply(lambda x:cleanResume(x))
resume_data.head()


# In[82]:


#getting the entire resume into text
corpus=" "
for i in range(0,len(resume_data)):
    corpus= corpus+ resume_data["cleaned"][i]
print(corpus)


# In[83]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
from wordcloud import WordCloud


# In[84]:


tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
#Tokenizing the text
tokens = tokenizer.tokenize(corpus)
len(tokens)


# In[85]:


words = []
# Looping through the tokens and make them lower case
for word in tokens:
    words.append(word.lower())
words[0:5]


# In[86]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
resume_data['new_Category'] = label.fit_transform(resume_data['Category'])
resume_data.head()


# In[87]:


plt.hist(resume_data['new_Category'])


# In[88]:


from sklearn.feature_extraction.text import TfidfVectorizer
text = resume_data['cleaned'].values
target = resume_data['new_Category'].values
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(text)
WordFeatures = word_vectorizer.transform(text)


# In[89]:


WordFeatures.shape


# # 3. Performing EDA

# In[90]:


category = resume_data['Category'].value_counts().reset_index()
category


# In[91]:


plt.figure(figsize=(12,8))
sns.barplot(x=category['Category'], y=category['index'], palette='cool')
plt.show()


# In[92]:


plt.figure(figsize=(12,8))
plt.pie(category['Category'], labels=category['index'],
        colors=sns.color_palette('cool'), autopct='%.0f%%')
plt.title('Category Distribution')
plt.show()


# # 4. Wordcloud

# In[93]:


stopwords = nltk.corpus.stopwords.words('english')
words_new = []

for word in words:
    if word not in stopwords:
        words_new.append(word)


# In[94]:


words_new[0:5]


# In[95]:


import nltk
nltk.download('omw-1.4')


# In[96]:


from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
wn = WordNetLemmatizer() 
lem_words=[]
for word in words_new:
    word=wn.lemmatize(word)
    lem_words.append(word)


# In[97]:


lem_words[0:5]


# In[98]:


same=0
diff=0
for n in range(0,2000):
    if(lem_words[n]==words_new[n]):
        same=same+1
    elif(lem_words[n]!=words_new[n]):
        diff=diff+1
print('Number of words Lemmatized=', diff)
print('Number of words not Lemmatized=', same)


# In[99]:


freq_dist = nltk.FreqDist(lem_words)
#Frequency Distribution Plot
plt.subplots(figsize=(20,12))
freq_dist.plot(30)


# In[100]:


res=' '.join([i for i in lem_words if not i.isdigit()])


# In[76]:


plt.subplots(figsize=(16,10))
wordcloud = WordCloud(
                          background_color='black',
                          max_words=200,
                          width=1400,
                          height=1200
                         ).generate(res)
plt.imshow(wordcloud)
plt.title('Resume Text WordCloud (200 Words)')
plt.axis('off')
plt.show()


# # 5. Training and Testing Data

# In[26]:


# Seprate train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(WordFeatures, target, random_state=24, test_size=0.2)


# In[28]:


# Model Training
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# In[29]:


# OneVsRest Classifier using KNeighbours
k_neighborus = OneVsRestClassifier(KNeighborsClassifier())
k_neighborus.fit(X_train, Y_train)
# Prediction
Y_pred_k_neighbour = k_neighborus.predict(X_test)


# In[30]:


print(f'---------------------------------\n| Training Accuracy   :- {(k_neighborus.score(X_train, Y_train)*100).round(2)}% |')
print(f'---------------------------------\n| Validation Accuracy :- {(k_neighborus.score(X_test, Y_test)*100).round(2)}% |\n---------------------------------')


# In[31]:


# OneVsRestClassifer using Random Forest
random_forest = OneVsRestClassifier(RandomForestClassifier())
random_forest.fit(X_train, Y_train)
# Prediction
Y_pred_random_forst = random_forest.predict(X_test)


# In[32]:


print(f'---------------------------------\n| Training Accuracy   :- {(random_forest.score(X_train, Y_train)*100).round(2)}% |')
print(f'---------------------------------\n| Validation Accuracy :- {(random_forest.score(X_test, Y_test)*100).round(2)}% |\n---------------------------------')


# In[33]:


# OneVsRestClassifer using Random Forest
random_forest = OneVsRestClassifier(RandomForestClassifier())
random_forest.fit(X_train, Y_train)
# Prediction
Y_pred_random_forst = random_forest.predict(X_test)


# In[34]:


# Metrics for K Neighbours
print(metrics.classification_report(Y_test, Y_pred_k_neighbour))


# In[35]:


# Metrics for Random Forest
print(metrics.classification_report(Y_test, Y_pred_random_forst))


# In[ ]:




