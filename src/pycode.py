#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
import sklearn.metrics
from sklearn import svm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


import nltk
# Word lemmatization
principal_data = pd.read_json("input/train.json")
principal_data['ingredients_clean_string'] = [' , '.join(z).strip() for z in principal_data['ingredients']]  
principal_data['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in principal_data['ingredients']]       


# In[3]:


principal_data.cuisine.value_counts()


# In[4]:


counters = {}
for cuisine in principal_data['cuisine'].unique():
    counters[cuisine] = Counter()
    indices = (principal_data['cuisine'] == cuisine)
    for ingredients in principal_data[indices]['ingredients']:
        counters[cuisine].update(ingredients)


# In[5]:


top15 = pd.DataFrame([[items[0] for items in counters[cuisine].most_common(15)] for cuisine in counters],
            index=[cuisine for cuisine in counters],
            columns=['top{}'.format(i) for i in range(1, 16)])
top15


# In[7]:


principal_data['all_ingredients'] = principal_data['ingredients'].map(";".join)
principal_data['all_ingredients']


# In[9]:


indices = principal_data['all_ingredients'].str.contains('sesame seeds')
principal_data[indices]['cuisine'].value_counts().plot(kind='bar',
                                                 title='sesame seeds found in one cuisine')


# In[ ]:


path = r'C:\Users\monda\Desktop\Spark\output\train_new.csv'
principal_data[[ 'id', 'ingredients_string' , 'cuisine' ]].to_csv(path, index=False)


# In[10]:


#Locating groups of similar cuisines
cuisine_ingredients = principal_data.groupby('cuisine').ingredients_string.sum()
cuisine_ingredients


# In[11]:


# examine the brazilian ingredients
cuisine_ingredients['brazilian'][0:500]


# In[13]:


vectorizer = TfidfVectorizer()
cuisine_dtm = vectorizer.fit_transform(cuisine_ingredients)
print(vectorizer.get_feature_names()[0:50])


# In[14]:


from sklearn import metrics
cuisine_similarity = []
for idx in range(cuisine_dtm.shape[0]):
    similarity = metrics.pairwise.cosine_similarity(cuisine_dtm[idx, :], cuisine_dtm).flatten()
    cuisine_similarity.append(similarity)


# In[15]:


# convert the results to a DataFrame
cuisine_list = cuisine_ingredients.index
cuisine_similarity = pd.DataFrame(cuisine_similarity, index=cuisine_list, columns=cuisine_list)
cuisine_similarity


# In[16]:


path = r'C:\Users\monda\Desktop\Spark\output\train_newcuisine_similarity_final.csv'
pd.DataFrame(cuisine_similarity, index=cuisine_list, columns=cuisine_list).to_csv(path, index=False)


# In[18]:


import seaborn as sns

# display the similarities as a heatmap
get_ipython().run_line_magic('matplotlib', 'inline')
sns.heatmap(cuisine_similarity)


# In[ ]:




