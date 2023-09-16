#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


# ### The Big Picture

# ##### After conducting an analysis of the available features in the dataset, it was discovered that the concept of lsoa holds a lot of potential. In essence, the lsoa are a geographical hierarchy that divide England and Whales into areas of similar populations. This is particularly interesting since these areas are then statistically studied across various indices of deprivation. These deprivation indices include metrics such as income, employment, and education. After carefully considering all the statistics reported on the LSOA, it was decided to use education deprivation as our metric of interest. This decision was based on the fact that education can play a major role in individuals' behavioural patterns. The natural consequent question would be **"does lower education result in more dangerous driving?"**
# 

# In[2]:


df_lsao = pd.read_csv('imd_eng.csv',encoding= 'unicode_escape')
df_lsao.head()


# In[3]:


df_lsao['edust_score'].describe()


# In[4]:


df_lsao['edust_rank'].describe()


# In[5]:


df_lsao.shape


#  #### An area's rank of Education Skills And Training Score is determined by taking into account both its performance (score) and how it stacks up against other locations throughout the country. According to the current ranking system, 1 represents the most deprived person, while 32,844 represents the least deprived person.

# In[6]:


df_lsao['edust_rank'].isnull().mean()*100


# In[7]:


df_lsao = df_lsao[['lsoacode','edust_rank']]
df_lsao['lsoacode'] = df_lsao['lsoacode'].astype(str)
df_lsao.head()


# In[8]:


df_lsao.shape


# In[9]:


df_lsao_wales = pd.read_csv('WIMD_Ranks.csv')
df_lsao_wales = df_lsao_wales[['Local Area (2011 LSOA)','Education']]
df_lsao_wales.rename(columns = {'Local Area (2011 LSOA)':'lsoacode','Education':'edust_rank'}, inplace = True)
df_lsao_wales.head()


# In[10]:


df_lsao.shape,df_lsao_wales.shape


# In[11]:


df_lsao = pd.concat([df_lsao,df_lsao_wales],axis=0)
df_lsao.head()


# In[12]:


df_lsao.shape


# In[13]:


def reverse_encode(df,column_name,look_up_table_df):
    df_look_table_column = look_up_table_df[look_up_table_df['column_name']==column_name]
    df_look_table_column.drop('column_name',axis =1,inplace=True)
    mapping_dict = dict()
    for index,row in df_look_table_column.iterrows():
        mapping_dict[row['Encoding']] = row['Value']
    print(df[column_name].replace(mapping_dict))
    df[column_name] = df[column_name].replace(mapping_dict)
    


# In[14]:


df_look_up = pd.read_csv('look_up_table.csv')
df_look_up_lsoa = df_look_up[df_look_up['column_name']=='lsoa_of_accident_location']
df_look_up_lsoa.drop('column_name',axis =1,inplace=True)
df_look_up_lsoa['Value'] = df_look_up_lsoa['Value'].astype(str)
df_look_up_lsoa.head()


# In[15]:


mapping_dict = dict()
for index,row in df_look_up_lsoa.iterrows():
    mapping_dict[row['Value']] = row['Encoding']
mapping_dict


# In[16]:


len(set(df_look_up_lsoa['Value']).difference(set(df_lsao['lsoacode'])))


# ## After retrieving all LSOA data available for both England and Wales, we still have 42 LSOA values whose data is not available.

# In[17]:


set(df_look_up_lsoa['Value']).difference(set(df_lsao['lsoacode']))


# In[18]:


df_look_up_lsoa[df_look_up_lsoa['Value'] == 'E01022070']['Value']


# In[19]:


set(df_look_up_lsoa['Value'])
    


# In[20]:


len(df_look_up_lsoa['Value'].unique())


# In[21]:


len(df_lsao['lsoacode'].unique())


# In[22]:


lsao_values = df_look_up_lsoa['Value'].unique()
df_lsao = df_lsao[df_lsao['lsoacode'].isin(lsao_values)]
df_lsao.rename(columns = {'edust_rank':'Education_Deprivation_Score'}, inplace = True)
df_lsao['lsoa_of_accident_location'] = df_lsao["lsoacode"].replace(mapping_dict)
df_education_index = df_lsao.drop('lsoacode',axis =1)
df_education_index.head()


# In[23]:


df_accidents = pd.read_csv('Final_Output.csv')
df_accidents.head()


# ### As discussed above, we have 42 LSOA values not available to us in the public datasets we were able to retrieve, this leads to 328 rows having missing values for the newly added column. since the rows with missing values acount for less than 1% of the dataset, we will opt to remove these rows. This will automatically be done by using inner join.

# In[24]:


df_accidents_augmented =df_accidents.merge(df_education_index,how ='inner')


# In[25]:


df_accidents_augmented.shape,df_accidents.shape


# In[46]:


df_accidents_augmented.shape[0]-df_accidents.shape[0]


# In[26]:


df_accidents_augmented.isnull().mean()*100


# In[27]:


df_accidents_augmented.head()


# In[28]:



df_accidents_augmented['Education_level'] = pd.cut(df_accidents_augmented['Education_Deprivation_Score'], 4,                            labels = ['Highly Deprived','Deprived','Educated','Highly Educated'])    

df_accidents_augmented.head()


# In[29]:


df_accidents_augmented_analysis = df_accidents_augmented.copy()


# In[30]:


reverse_encode(df_accidents_augmented_analysis,'accident_severity',df_look_up)


# In[31]:


df_accidents_augmented_analysis.head()


# ## does education level affect accident severity?

# In[32]:


x,y = 'Education_level', 'accident_severity'

ax =(df_accidents_augmented_analysis
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
plt.xticks(rotation = 90)
plt.yticks(np.arange(0, 100, 5.0))
plt.show()


# In[ ]:





# In[33]:


df_accidents_augmented_analysis.groupby(x)[y].value_counts(normalize=True).mul(100)


# In[34]:


sns.heatmap(df_accidents_augmented_analysis.corr())


# ## does less education lead to more casualties?

# In[35]:


x,y = 'Education_level', 'number_of_casualties'

sns.barplot(data=df_accidents_augmented_analysis, x=x, y=y)


# In[36]:


df_accidents_augmented_analysis.groupby(x)[y].mean()


# ## After analyzing the data from both questions, it was discovered that the education level for each LSOA, does not have a significant impact on either accident severity or number of casualties.
# 

# ## scaling the new column

# In[37]:


sns.kdeplot(df_accidents_augmented['Education_Deprivation_Score'])


# In[38]:


df_accidents_augmented['Education_Deprivation_Score'].skew()


# ## Feature Scaling 

# In[39]:


# For min_max scaling
from sklearn.preprocessing import MinMaxScaler
education_score_scaled = MinMaxScaler().fit_transform(df_accidents_augmented[['Education_Deprivation_Score']]) 
df_accidents_augmented['Education_Deprivation_Score'] = education_score_scaled


# In[40]:


df_accidents_augmented.head()


# In[41]:


df_accidents_augmented.Education_level.unique()


# In[42]:


df_accidents_augmented.Education_level.isnull().mean()*100


# In[43]:


def LabelEncode(dataframe,column_name,mapping={}):
    le_namemapping ={}
    look_up = {}
    df = dataframe.copy()
    if(len(mapping)!=0):
        df[column_name] = df[column_name].map(mapping)
        look_up = {
            'column_name': column_name,
            'Value': mapping.keys(),
            'Encoding': mapping.values()
        }
    else:
        le = preprocessing.LabelEncoder()
        le.fit(df[column_name])
        
        le_namemapping = dict(zip(le.classes_, le.transform(le.classes_)))
        look_up = {
            'column_name': column_name,
            'Value': le.classes_ ,
            'Encoding': le.transform(le.classes_)
        }
        df[column_name] = df[column_name].map(le_namemapping)
    look_up = pd.DataFrame(look_up)
    return df,look_up  


# In[44]:


mappings = {
        'Highly Deprived':1,
        'Deprived':2,
        'Educated':3,
        'Highly Educated':4
    }

df_accidents_augmented,mappings = LabelEncode(df_accidents_augmented,'Education_level',mappings)
df_accidents_augmented['Education_level'].value_counts()


# In[45]:


df_look_up = pd.concat([df_look_up,mappings],axis = 0)
df_look_up


# In[ ]:


df_look_up.to_csv('look_up_table.csv',index=False)

