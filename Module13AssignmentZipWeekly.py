import numpy as np
import os
import pandas as pd
import csv
import geopandas
import rtree
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from shapely.geometry import Point

###Set the path
os.getcwd()
os.chdir(r'C:\Users\Shinji\Module13\Assignmnet')

###Create a Function that calculate the RF for any zipcode in Brooklyn

def RF_NYCzipNumber(zipNumber):
    ###Load csv file
    df = pd.read_csv('Complain2010_2015_Brooklyn.csv')

    ###Rename complaint type column
    df.rename(columns={'Complaint Type': 'complaint_type'}, inplace=True)

    ###Define all posible labels from Brooklyn
    labels = [[df.complaint_type.unique()]]

    ###
    Zip11204 = df['Incident Zip'] == zipNumber
    df = df[Zip11204]
    ###Drop columns
    df1 = df[['Unique Key','Created Date','complaint_type','Incident Zip','Latitude','Longitude']]


    ###Create date and year column
    df1['Year'] = df1['Created Date'].astype(str).str[6:10]
    df1['Date'] = df['Created Date'].str[:10]

    ###Pivot table based on date and complains
    df2 = df1.pivot_table(index='Date',columns='complaint_type', values='Incident Zip',aggfunc='count')
    # print(df2.head(5))
    df2.fillna(0,inplace=True)
    # print(df2.head(5))

    ##################Load the crime files and transform it into shapes
    crime = pd.read_csv('Crime2010_2015Brooklyn.csv')

    crime = crime.dropna(subset=['Longitude','Latitude'])

    ####Create the geometry column
    crime['geometry'] = crime.apply(lambda x: Point((float(x.Longitude), float(x.Latitude))), axis=1)
    # print(crime.dtypes)

    crime = crime[['CMPLNT_FR_DT','PD_DESC','KY_CD','geometry']]

    ####Create the geometry in geopandas
    crime = geopandas.GeoDataFrame(crime, geometry='geometry')

    crime.crs= "+proj=longlat +datum=WGS84 +no_defs"

    ####Call the shapefile of NYZ zipcodes
    zipshape = geopandas.read_file('ZipCodeNYC.shp')

    zipshape.crs= "+proj=longlat +datum=WGS84 +no_defs"

    ###Subset by zipcode
    zipshape.ZIPCODE = pd.to_numeric(zipshape.ZIPCODE,errors='coerce')
    zip11204 = zipshape['ZIPCODE'] == zipNumber
    zipshape1 = zipshape[zip11204]

    ### Make the spatial JoiN
    crimezip1 = geopandas.sjoin(zipshape1,crime, how="inner", op='intersects')
    #########################

    ### Create variables to merge the tables
    crimezip1['Year'] = crimezip1['CMPLNT_FR_DT'].astype(str).str[6:10]

    crimezip2 = crimezip1[['CMPLNT_FR_DT','PD_DESC','ZIPCODE','Year']]

    crimezip = crimezip2.pivot_table(index='CMPLNT_FR_DT',columns='Year', values='ZIPCODE',aggfunc='count')
    
    crimezip.rename(columns={'2010': 'Crimes10_14','2011': 'Crimes2011','2012': 'Crimes2012','2013': 'Crimes2013','2014': 'Crimes2014','2015': 'Crimes2015'}, inplace=True)

    crimezip.fillna(0,inplace=True)
    ###Merge the tables
    df3 = pd.merge(crimezip, df2, left_index=True, right_index=True)

    ###Create year columns
    df3.reset_index(level=0, inplace=True)
    df3['Year'] = df3['index'].str[6:10]
    ###Create columns with crimes per year
    col_list = ['Crimes10_14','Crimes2011','Crimes2012','Crimes2013','Crimes2014']

    ###Create a combinded column of crimes
    df3['Crimes10_14'] = df3[col_list].sum(axis=1)

    df3 = df3.drop('Crimes2011', 1)
    df3 = df3.drop('Crimes2012', 1)
    df3 = df3.drop('Crimes2013', 1)
    df3 = df3.drop('Crimes2014', 1)

    #####Create the week days and WeekYear Index
    df3['Date'] = pd.to_datetime(df3['index'])
    df3['week'] = df3['Date'].dt.week.astype('str')
    df3['YearWeek'] = df3['Year'].map(str) + df3['week']
    df3 = df3.drop('Date', 1)
    df3 = df3.drop('week', 1)

    df3 = df3.groupby('YearWeek').agg('sum')

    df3.reset_index(level=0, inplace=True)

    df3['Year'] = df3['YearWeek'].str[0:4]

    ###Convert from pandas to numpy array
    df2014 = df3['Year'] != '2015'
    m1 = np.asarray(df3[df2014])

    df2015 = df3['Year'] == '2015'
    m2 = np.asarray(df3[df2015])

    ###Define train and test arrays
    xtrain = m1[:, 3:-1]
    ytrain = m1[:, 1]

    xtest = m2[:, 3:-1]
    ytest = m2[:, 2]

    ###Define the first regression
    rf = RandomForestRegressor(n_estimators=100, criterion="mse", max_depth=None, max_features="auto", bootstrap=True,
     min_samples_split=2, n_jobs=1)

    labels = list(df3.columns.values[3:-1])

    #####Train and Predicts
    rf.fit(xtrain, ytrain)
    ypred = rf.predict(xtest)

    ####First result of the regression
    print('Weekly Results of first RF regression')
    print("RF (the number of trees in the forest is 100): ", r2_score(ytest, ypred))
    ####Most important variables
    zipped_rb = list (zip (labels, rf.feature_importances_))
    print(zipped_rb)
    ######################END FIRST RF REGRESSION
    c = list (zip (labels, rf.feature_importances_))

    ###Define new variables
    new_labels=[]
    for i in c:
        if i[1]>=0.04:
            new_labels.append(i[0])

    new_labels.append('Year')
    new_labels.insert(0,'Crimes2015')
    new_labels.insert(0,'Crimes10_14')
    new_labels.insert(0,'YearWeek')

    ###Subset the new variables from data frame
    df3 = df3[new_labels]

    ###Convert from pandas to numpy array
    df2014 = df3['Year'] != '2015'
    m1 = np.asarray(df3[df2014])

    df2015 = df3['Year'] == '2015'
    m2 = np.asarray(df3[df2015])

    ###Define train and test arrays
    xtrain = m1[:, 3:-1]
    ytrain = m1[:, 1]

    xtest = m2[:, 3:-1]
    ytest = m2[:, 2]

    ###Define the second regression
    rf = RandomForestRegressor(n_estimators=100, criterion="mse", max_depth=None, max_features="auto", bootstrap=True,
                               min_samples_split=2, n_jobs=1)

    ####Second result of the regression, most important variables
    rf.fit(xtrain, ytrain)
    ypred = rf.predict(xtest)

    print('Weekly Results of second RF regression with most important features')
    print("RF Important Variables (the number of trees in the forest is 100): ", r2_score(ytest, ypred))

    new_labels = new_labels[3:-1]
    ####Most important variables second RF regression
    zipped_rb = list (zip (new_labels, rf.feature_importances_))
    print(zipped_rb)


###Call the function
RF_NYCzipNumber(11204)

######################################################
#####################################################
###############END################################
