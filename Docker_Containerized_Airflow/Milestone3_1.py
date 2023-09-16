import pandas as pd
import numpy as np
import datetime
from scipy import stats
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

def getOptimalzCutOff(zScores,threshold):
    for i in range(1,10):
        filtered_entries = zScores < i
        percentage_dropped =100 - ((sum(filtered_entries)/len(filtered_entries))*100)
        if(percentage_dropped <= threshold):
            return filtered_entries

def OneHotEncode(dataframe,column_name,column_to_drop = None,drop_first = False):
    df = dataframe.copy()
    if(drop_first):
        encoded = pd.get_dummies(df[column_name],prefix = column_name,drop_first = drop_first)
    else:
        encoded = pd.get_dummies(df[column_name],prefix = column_name)
    if(column_to_drop != None):
        encoded.drop(str(column_name)+'_'+column_to_drop, axis = 1, inplace=True)
    df.drop(column_name,axis =1, inplace = True)
    df = pd.concat([df,encoded],axis = 1)
    
    
    
    
    return df;

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

def get_month(date):
    month = date.strftime("%m")
    return month

def get_day(date):
    month = date.strftime("%d")
    return month

def get_hour(time):
    hours, minutes = map(int, time.split(':'))
    return hours

def get_minutes(time):
    hours, minutes = map(int, time.split(':'))
    return minutes

def read_file(file_name):
    file_accidents = file_name
    df_accidents = pd.read_csv(file_accidents,index_col=0)
    df_accidents['date'] = pd.to_datetime(df_accidents['date'])
    return df_accidents

def clean_operation(df_accidents):
    df_accidents_complete = df_accidents.copy()
    df_accidents_complete['second_road_number'] = df_accidents.second_road_number.fillna(-1)
    f = lambda x: x.mode().values[0] if  len(x.mode().values) > 0 else np.nan
    df_accidents_complete['road_type'] = df_accidents_complete['road_type'].fillna(df_accidents_complete.groupby(['first_road_class'])['road_type'].transform(f))

    f = lambda x: x.mode().values[0] if  len(x.mode().values) > 0 else np.nan
    df_accidents_complete['weather_conditions'] = df_accidents_complete['weather_conditions'].fillna(df_accidents_complete.groupby(['road_surface_conditions'])['weather_conditions'].transform(f))

    df_accidents_complete['light_conditions'].replace('Darkness - lighting unknown', np.nan, inplace = True)

    df_accidents_daylight_dropped = df_accidents_complete.copy()
    df_accidents_daylight_dropped = df_accidents_daylight_dropped[df_accidents_daylight_dropped['light_conditions'] != 'Daylight']

    df_accidents_light_nandrop = df_accidents_daylight_dropped.dropna()

    for index,row in df_accidents_complete.iterrows():
        if pd.isnull(row['light_conditions']):
            random_sample = df_accidents_light_nandrop.sample()
            df_accidents_complete.loc[index, 'light_conditions'] = random_sample['light_conditions'][0]

    df_accidents_complete['road_surface_conditions'].replace('Data missing or out of range', np.nan, inplace = True)

    f = lambda x: x.mode().values[0] if  len(x.mode().values) > 0 else np.nan
    df_accidents_complete['road_surface_conditions'] = df_accidents_complete['road_surface_conditions'].fillna(df_accidents_complete.groupby(['weather_conditions'])['road_surface_conditions'].transform(f))

    clf = LogisticRegression()
    df_Xtrain = df_accidents_complete[df_accidents_complete.trunk_road_flag != 'Data missing or out of range']
    X_train = df_Xtrain[['speed_limit','road_type']]
    encoded_road_type = pd.get_dummies(X_train.road_type, drop_first = True)
    mappings = {
            20:0,
            30:1,
            40:2,
            50:3,
            60:4,
            70:5
        }
    X_train['speed_limit'] = X_train['speed_limit'].map(mappings)
    X_train = X_train.drop('road_type', axis = 1)   
    X_train = pd.concat([X_train,encoded_road_type],axis=1)
    y_train =  pd.get_dummies(df_Xtrain.trunk_road_flag, drop_first = True)

    df_Xtest = df_accidents_complete[df_accidents_complete.trunk_road_flag == 'Data missing or out of range']
    X_test = df_Xtest[['speed_limit','road_type']]
    X_test = df_Xtest[['speed_limit','road_type']]
    encoded_road_type = pd.get_dummies(X_test.road_type, drop_first = True)
    X_test['speed_limit'] = X_test['speed_limit'].map(mappings)
    X_test = X_test.drop('road_type', axis = 1)   
    X_test = pd.concat([X_test,encoded_road_type],axis=1)
    X_train = X_train.values.tolist()
    y_train = y_train.values.tolist()
    clf.fit(X_train,y_train)

    for index,row in df_accidents_complete.iterrows():
        if(row.trunk_road_flag == 'Data missing or out of range'):
            prediction = clf.predict([X_test.loc[index]])[0]
            df_accidents_complete.loc[index,'trunk_road_flag'] = 'Trunk (Roads managed by Highways England)' if prediction == 0 else 'Non-trunk' 
        

    df_accidents_complete['lsoa_of_accident_location'] = df_accidents_complete.lsoa_of_accident_location.replace([-1,'-1'], np.nan)
    sum(df_accidents_complete['lsoa_of_accident_location'].isnull())
    df_imputed = df_accidents_complete.copy()
    df_imputed['approximate_latitude'] = pd.qcut(df_imputed['latitude'], q=10, precision=0)
    df_imputed['approximate_longitude'] = pd.qcut(df_imputed['longitude'], q=10, precision=0)

    f = lambda x: x.mode().values[0] if  len(x.mode().values) > 0 else np.nan
    df_imputed['lsoa_of_accident_location'] = df_imputed['lsoa_of_accident_location'].fillna(df_imputed.groupby(['approximate_latitude','approximate_longitude'])['lsoa_of_accident_location'].transform(f))
    df_accidents_complete['lsoa_of_accident_location'] = df_imputed['lsoa_of_accident_location']

    df_accidents_complete['local_authority_highway'].replace('-1', np.nan, inplace = True)
    f = lambda x: x.mode().values[0] if  len(x.mode().values) > 0 else np.nan
    df_accidents_complete['local_authority_highway'] = df_accidents_complete['local_authority_highway'].fillna(df_accidents_complete.groupby(['police_force'])['local_authority_highway'].transform(f))

    df_accidents_complete.drop_duplicates(set(df_accidents) - set(['accident_reference','accident_index']),inplace=True)

    df_accidents_complete.drop(labels=['accident_year','accident_reference'], axis=1, inplace=True)

    df_accidents_complete.drop(labels=['location_easting_osgr','location_northing_osgr'], axis=1, inplace=True)
    zcas = np.abs(stats.zscore(df_accidents_complete.number_of_casualties))
    filtered_entries_cas = getOptimalzCutOff(zcas,1)
    df_cleaned_filtered_outliers = df_accidents_complete[filtered_entries_cas]
    zcar = np.abs(stats.zscore(df_cleaned_filtered_outliers.number_of_vehicles))
    filtered_entries_car = getOptimalzCutOff(zcar,1)
    df_cleaned_filtered_outliers = df_cleaned_filtered_outliers[filtered_entries_car]
    return df_cleaned_filtered_outliers

def transformation_operation (df_cleaned_filtered_outliers):
    df_cleaned_filtered_outliers['Week number'] = df_cleaned_filtered_outliers['date'].dt.strftime("%U").astype(int)
    df_cleaned_filtered_outliers['date'] = df_cleaned_filtered_outliers['date'].dt.date
    Values = []
    Encodings =[]
    df_cleaned_filtered_outliers['Week number'] =df_cleaned_filtered_outliers['Week number'].astype(int)
    num_weeks = df_cleaned_filtered_outliers['Week number'].max()
    for weeknumber in range(0,num_weeks+1):
        dates = df_cleaned_filtered_outliers[df_cleaned_filtered_outliers['Week number']== weeknumber].date.unique()
        if(len(dates)!=0):
            dates.sort()
            Values.append(str(dates[0]) + ' - ' +str(dates[len(dates)-1]))
            Encodings.append(weeknumber)
    look_up = {
        'column_name': 'Week num',
        'Value': Values,
        'Encoding': Encodings
        
    }

    look_up_table = pd.DataFrame()
    look_up_table = pd.concat([look_up_table,pd.DataFrame(look_up)])
    df_cleaned_filtered_outliers_encoded = df_cleaned_filtered_outliers.copy()
    mappings = {
            'Slight':1,
            'Serious':2,
            'Fatal':3
        }

    df_cleaned_filtered_outliers_encoded,mappings = LabelEncode(df_cleaned_filtered_outliers_encoded,'accident_severity',mappings)
    look_up_table = pd.concat([look_up_table,mappings],axis = 0)

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'first_road_class','Unclassified')

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'second_road_class','-1')

    mappings = {
            'Non-trunk':0,
            'Trunk (Roads managed by Highways England)':1,
        }
    df_cleaned_filtered_outliers_encoded,mappings = LabelEncode(df_cleaned_filtered_outliers_encoded,'trunk_road_flag',mappings)
    look_up_table = pd.concat([look_up_table,mappings],axis = 0)
    mappings = {
            'No':0,
            'Yes':1,
        }
    df_cleaned_filtered_outliers_encoded,mappings = LabelEncode(df_cleaned_filtered_outliers_encoded,'did_police_officer_attend_scene_of_accident',mappings)
    look_up_table = pd.concat([look_up_table,mappings],axis = 0)

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'day_of_week',drop_first=True)

    mappings = {
            'Rural':0,
            'Urban':1,
        }
    df_cleaned_filtered_outliers_encoded,mappings = LabelEncode(df_cleaned_filtered_outliers_encoded,'urban_or_rural_area',mappings)
    df_cleaned_filtered_outliers_encoded.rename(columns = {'urban_or_rural_area':'is_Urban'}, inplace = True)
    look_up_table = pd.concat([look_up_table,mappings],axis = 0)

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'road_surface_conditions',drop_first=True)

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'road_type',drop_first=True)

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'light_conditions',column_to_drop='Daylight')

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'weather_conditions',column_to_drop="Other")

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'junction_detail',column_to_drop='Other junction')

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'junction_control',column_to_drop='Data missing or out of range')

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'pedestrian_crossing_human_control',column_to_drop='None within 50 metres ')

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'pedestrian_crossing_physical_facilities',column_to_drop='No physical crossing facilities within 50 metres')

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'special_conditions_at_site',column_to_drop='None')

    df_cleaned_filtered_outliers_encoded = OneHotEncode(df_cleaned_filtered_outliers_encoded,'carriageway_hazards',column_to_drop='None')

    df_cleaned_filtered_outliers_encoded,mappings = LabelEncode(df_cleaned_filtered_outliers_encoded,'police_force')

    look_up_table = pd.concat([look_up_table,mappings],axis = 0)

    df_cleaned_filtered_outliers_encoded,mappings = LabelEncode(df_cleaned_filtered_outliers_encoded,'local_authority_district')

    df_cleaned_filtered_outliers_encoded,mappings = LabelEncode(df_cleaned_filtered_outliers_encoded,'local_authority_ons_district')

    look_up_table = pd.concat([look_up_table,mappings],axis = 0)

    df_cleaned_filtered_outliers_encoded,mappings = LabelEncode(df_cleaned_filtered_outliers_encoded,'local_authority_highway')

    look_up_table = pd.concat([look_up_table,mappings],axis = 0)

    df_cleaned_filtered_outliers_encoded = df_cleaned_filtered_outliers_encoded.replace('first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ',0)

    df_cleaned_filtered_outliers_encoded['first_road_number'] = df_cleaned_filtered_outliers_encoded.first_road_number.astype(float)
    df_cleaned_filtered_outliers_encoded['second_road_number'] = df_cleaned_filtered_outliers_encoded.second_road_number.astype(float)
    df_cleaned_filtered_outliers_encoded['first_road_number'] = df_cleaned_filtered_outliers_encoded.first_road_number.astype(int)
    df_cleaned_filtered_outliers_encoded['second_road_number'] = df_cleaned_filtered_outliers_encoded.second_road_number.astype(int)

    df_cleaned_filtered_outliers_encoded,mappings = LabelEncode(df_cleaned_filtered_outliers_encoded,'lsoa_of_accident_location')
    look_up_table = pd.concat([look_up_table,mappings],axis = 0)

    df_cleaned_filtered_outliers_encoded,mappings = LabelEncode(df_cleaned_filtered_outliers_encoded,'speed_limit')
    look_up_table = pd.concat([look_up_table,mappings],axis = 0)

    df_cleaned_filtered_outliers_encoded_normalised_augmented = df_cleaned_filtered_outliers_encoded.copy()
    df_cleaned_filtered_outliers_encoded_normalised_augmented['is_Weekend'] = (df_cleaned_filtered_outliers_encoded_normalised_augmented['day_of_week_Saturday'] == 1) | (df_cleaned_filtered_outliers_encoded_normalised_augmented['day_of_week_Sunday'] == 1)
    df_cleaned_filtered_outliers_encoded_normalised_augmented['is_Weekend'] = df_cleaned_filtered_outliers_encoded_normalised_augmented['is_Weekend'].astype(int)

    df_cleaned_filtered_outliers_encoded_normalised_augmented['is_Dark_and_Raining'] = ((df_cleaned_filtered_outliers_encoded_normalised_augmented['weather_conditions_Raining + high winds'] == 1)                                                                           | (df_cleaned_filtered_outliers_encoded_normalised_augmented['weather_conditions_Raining no high winds'] == 1))                                                                            & ((df_cleaned_filtered_outliers_encoded_normalised_augmented['light_conditions_Darkness - lights unlit'] == 1)                                                                               | (df_cleaned_filtered_outliers_encoded_normalised_augmented['light_conditions_Darkness - no lighting'] == 1))

    df_cleaned_filtered_outliers_encoded_normalised_augmented['Month'] = df_cleaned_filtered_outliers_encoded_normalised_augmented['date'].apply(get_month).astype(int)
    df_cleaned_filtered_outliers_encoded_normalised_augmented['Day'] = df_cleaned_filtered_outliers_encoded_normalised_augmented['date'].apply(get_day).astype(int)
    df_cleaned_filtered_outliers_encoded_normalised_augmented['Hour'] = df_cleaned_filtered_outliers_encoded_normalised_augmented['time'].apply(get_hour)
    df_cleaned_filtered_outliers_encoded_normalised_augmented['Minutes'] = df_cleaned_filtered_outliers_encoded_normalised_augmented['time'].apply(get_minutes)
    df_final = df_cleaned_filtered_outliers_encoded_normalised_augmented.drop(['date','time'],axis =1)
    return df_final, look_up_table

def load_operation(df_final, look_up_table):
    look_up_table.to_csv('look_up_table.csv',index=False)
    df_final.to_csv('cleaned_transformed.csv',index=False)

def clean_and_transform(file_name):
    df_accidents = read_file(file_name)
    df_cleaned_filtered_outliers = clean_operation(df_accidents)
    df_final, look_up_table = transformation_operation (df_cleaned_filtered_outliers)
    load_operation(df_final, look_up_table)