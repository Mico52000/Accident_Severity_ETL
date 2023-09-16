#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import os



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


def reverse_encode(df,column_name,look_up_table_df):
    df_look_table_column = look_up_table_df[look_up_table_df['column_name']==column_name]
    df_look_table_column.drop('column_name',axis =1,inplace=True)
    mapping_dict = dict()
    for index,row in df_look_table_column.iterrows():
        mapping_dict[row['Encoding']] = row['Value']
    ##print(df[column_name].replace(mapping_dict))
    df[column_name] = df[column_name].replace(mapping_dict)

def read_external_source(source_list):
    df_lsao = pd.read_csv(source_list[0],encoding= 'unicode_escape')
    df_lsao = df_lsao[['lsoacode','edust_rank']]
    df_lsao['lsoacode'] = df_lsao['lsoacode'].astype(str)
    df_lsao_wales = pd.read_csv(source_list[1])
    df_lsao_wales = df_lsao_wales[['Local Area (2011 LSOA)','Education']]
    df_lsao_wales.rename(columns = {'Local Area (2011 LSOA)':'lsoacode','Education':'edust_rank'}, inplace = True)
    df_lsao = pd.concat([df_lsao,df_lsao_wales],axis=0)
    return df_lsao

def read_look_up(look_up_table_name):
    ##df_look_up = pd.read_csv('look_up_table.csv')
    df_look_up = pd.read_csv(look_up_table_name)
    df_look_up_lsoa = df_look_up[df_look_up['column_name']=='lsoa_of_accident_location']
    df_look_up_lsoa.drop('column_name',axis =1,inplace=True)
    df_look_up_lsoa['Value'] = df_look_up_lsoa['Value'].astype(str)

    return df_look_up,df_look_up_lsoa
# ## After retrieving all LSOA data available for both England and Wales, we still have 42 LSOA values whose data is not available.


def integrate_sources(cleaned_transformed_file,df_lsao,df_look_up,df_look_up_lsoa):
    mapping_dict = dict()
    for _,row in df_look_up_lsoa.iterrows():
        mapping_dict[row['Value']] = row['Encoding']
    lsao_values = df_look_up_lsoa['Value'].unique()
    df_lsao = df_lsao[df_lsao['lsoacode'].isin(lsao_values)]
    df_lsao.rename(columns = {'edust_rank':'Education_Deprivation_Score'}, inplace = True)
    df_lsao['lsoa_of_accident_location'] = df_lsao["lsoacode"].replace(mapping_dict)
    df_education_index = df_lsao.drop('lsoacode',axis =1)



    df_accidents = pd.read_csv(cleaned_transformed_file)

    df_accidents_augmented =df_accidents.merge(df_education_index,how ='inner')
    df_accidents_augmented['Education_level'] = pd.cut(df_accidents_augmented['Education_Deprivation_Score'], 4,labels = ['Highly Deprived','Deprived','Educated','Highly Educated'])    
    df_accidents_augmented_analysis = df_accidents_augmented.copy()
    reverse_encode(df_accidents_augmented_analysis,'accident_severity',df_look_up)
    education_score_scaled = MinMaxScaler().fit_transform(df_accidents_augmented[['Education_Deprivation_Score']]) 
    df_accidents_augmented['Education_Deprivation_Score'] = education_score_scaled
    mappings = {
            'Highly Deprived':1,
            'Deprived':2,
            'Educated':3,
            'Highly Educated':4
        }

    df_accidents_augmented,mappings = LabelEncode(df_accidents_augmented,'Education_level',mappings)
    df_look_up = pd.concat([df_look_up,mappings],axis = 0)
    return df_accidents_augmented,df_look_up


def load_augmeted(df_accidents_augmented,df_look_up,df_final_name,look_up_table_name):
    df_look_up.to_csv(look_up_table_name,index=False)
    df_accidents_augmented.to_csv(df_final_name,index=False)

def augment_dataset(sourcelist,look_up_table,cleaned_transformed,look_up_final,augmented_data):
    if os.path.isfile(augmented_data) and os.path.isfile(look_up_final):
        print("The lookup table and the augmented file already exist and are saved as csv.")
    else:
        df_lsao = read_external_source(sourcelist)
        df_look_up,df_look_up_lsoa= read_look_up(look_up_table)
        df_accidents_augmented,df_look_up = integrate_sources(cleaned_transformed,df_lsao,df_look_up,df_look_up_lsoa)
        load_augmeted(df_accidents_augmented,df_look_up,augmented_data,look_up_final)