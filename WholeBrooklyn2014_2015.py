import numpy as np
import os
import pandas as pd
import csv
# import geopandas
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sodapy import Socrata

os.getcwd()
os.chdir(r'C:\Users\Shinji\Module13\Assignmnet')

###Load csv file
df = pd.read_csv('WholeBrooklyn2014_2015Complain.csv')
print(df.dtypes)


###Drop columns
df1 = df[['Unique Key','Created Date','Complaint Type','Incident Zip','Latitude','Longitude']]
df1['Date'] = pd.to_datetime(df['Created Date'].str[:10])

####Create the pivot table with the complains
df2 = df1.pivot_table(index='Date',columns='Complaint Type', values='Incident Zip',aggfunc='count')

df2.fillna(0,inplace=True)

####Load the Crime Table Mixed with the compalin
Rf_df = pd.read_csv('WholeBrooklynCrime_complaint2014.csv')
Rf_df1 = pd.read_csv('WholeBrooklynCrime_complaint2015.csv')
m1 = Rf_df.values
m2 = Rf_df.values

######Declare the test and trainning from 2014 and 2015
xtrain = m1[:, 2:]
ytrain = m1[:, 1]

xtest = m2[:, 2:]
ytest = m2[:, 1]

######Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, criterion="mse", max_depth=None, max_features="auto", bootstrap=True,
                           min_samples_split=2, n_jobs=1)

######Labels
labels = ['Animal Abuse','BEST/Site Safety','Bike Rack Condition','Bridge Condition','Curb Condition','DOOR/WINDOW','Dead Tree','Derelict Bicycle','Derelict Vehicle','Derelict Vehicles','Dirty Conditions','ELECTRIC','Electrical','FLOORING/STAIRS','Graffiti','Highway Condition','Homeless Encampment','Industrial Waste','Litter Basket / Request','Maintenance or Facility','Missed Collection (All Materials)','Overflowing Litter Baskets','Overgrown Tree/Branches','PAINT/PLASTER',
            'Root/Sewer/Sidewalk Condition','SAFETY','Sanitation Condition','Sidewalk Condition','Street Condition','Street Light Condition','Street Sign - Damaged','Street Sign - Dangling','Street Sign - Missing','Sweeping/Inadequate','Sweeping/Missed','Traffic Signal Condition','UNSANITARY CONDITION','Unsanitary Animal Pvt Property','Vacant Lot','WATER LEAK']


#####Fit the model
rf.fit(xtrain, ytrain)
ypred = rf.predict(xtest)
print(ypred)
print("RF: ", r2_score(ytest, ypred))

zipped_rf = list (zip (labels, rf.feature_importances_))
print(zipped_rf)

# wr = csv.writer(test, quoting=csv.QUOTE_ALL)
#     wr.writerow(mylist)

# Apply GBR
gb = GradientBoostingRegressor(n_estimators=100, loss="ls", learning_rate=0.1, criterion="friedman_mse")
gb.fit(xtrain, ytrain)
ypred = gb.predict(xtest)
print("GB: ", r2_score(ytest, ypred))

zipped_gb = list (zip (labels, gb.feature_importances_))
print(zipped_gb)



######################################################
#####################################################
###############TESTING################################
