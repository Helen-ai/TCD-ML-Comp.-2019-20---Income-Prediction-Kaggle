# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import category_encoders as ce
%matplotlib inline

def main():
    train_csv = '/kaggle/input/tcd-ml-comp-201920-income-pred-group/tcd-ml-1920-group-income-train.csv'
    test_csv = '/kaggle/input/tcd-ml-comp-201920-income-pred-group/tcd-ml-1920-group-income-test.csv'
    train_dataset = pd.read_csv(train_csv) 
    test_dataset = pd.read_csv(test_csv)
    
    #delDuplicates(train_dataset)

    #handle inconsistent data
    train_dataset = findUniqueVals(train_dataset)
    test_dataset = findUniqueVals(test_dataset)
    
    #dropping nan in Gender improved mae as compared to filling it with unknown or the mode (male)
    train_dataset.dropna(subset=['Gender'], inplace = True)

    #handle missing data
    train_dataset = replaceNaN(train_dataset)
    train_dataset['Total Yearly Income [EUR]'] = train_dataset['Total Yearly Income [EUR]'].fillna(int(train_dataset['Total Yearly Income [EUR]'].mean()))
    test_dataset = replaceNaN(test_dataset)
    
    train_dataset = train_dataset.drop(columns=['Profession','Instance'])
    test_dataset = test_dataset.drop(columns=['Profession','Instance'])
    
    x = train_dataset.drop(columns=['Total Yearly Income [EUR]']) 
    y = train_dataset['Total Yearly Income [EUR]']
    y = y.apply(np.log)
    
    #split data
    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=0)

    #target encode categorical variables
    enc = ce.TargetEncoder().fit(x_train, y_train)
    x_train = enc.transform(x_train)
    x_test = enc.transform(x_test)

    #train model
    model=CatBoostRegressor(iterations=15000, depth=7, learning_rate=0.15, loss_function='MAE',
                            verbose=100, od_type="Iter", od_wait=100)
    model.fit(x_train, y_train, eval_set=(x_test, y_test))

    #make predictions
    y_predicted = model.predict(x_test)

    #calculate error
    print("mae: %.2f" % mean_absolute_error(np.exp(y_test), np.exp(y_predicted)))
    
    submission(test_dataset,model,enc)
    
def submission(test_dataset,model,enc):
    y = test_dataset['Total Yearly Income [EUR]']
    x = test_dataset.drop(columns=['Total Yearly Income [EUR]'])
    x = enc.transform(x)
    y_pred = model.predict(x)
    y_pred = np.exp(y_pred)
    
    submis = pd.read_csv('/kaggle/input/tcd-ml-comp-201920-income-pred-group/tcd-ml-1920-group-income-submission.csv')
    submis['Total Yearly Income [EUR]'] = y_pred

    filename = 'Income Predictions.csv'
    submis.to_csv(filename,index=False)
    print('Saved file: ' + filename)

def replaceNaN(df): 
    df['Gender'] = df['Gender'].fillna('male') #mode
    df['University Degree'] = df['University Degree'].fillna('UNKNOWN')
    df['Profession'] = df['Profession'].fillna('UNKNOWN')
    df['Housing Situation'] = df['Housing Situation'].fillna('UNKNOWN')
    df['Country'] = df['Country'].fillna('UNKNOWN')
    df['Satisfation with employer'] = df['Satisfation with employer'].fillna('UNKNOWN')
    df['Hair Color'] = df['Hair Color'].fillna('UNKNOWN')
        
    df['Year of Record'] = df['Year of Record'].fillna(df['Year of Record'].mean())
    df['Crime Level in the City of Employement'] = df['Crime Level in the City of Employement'].fillna(df['Crime Level in the City of Employement'].mean())
    df['Work Experience in Current Job [years]'] = df['Work Experience in Current Job [years]'].fillna(df['Work Experience in Current Job [years]'].mean())
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Size of City'] = df['Size of City'].fillna(df['Size of City'].mean())
    df['Body Height [cm]'] = df['Body Height [cm]'].fillna(df['Body Height [cm]'].mean())
    df['Yearly Income in addition to Salary (e.g. Rental Income)'] = df['Yearly Income in addition to Salary (e.g. Rental Income)'].fillna(df['Yearly Income in addition to Salary (e.g. Rental Income)'].mean())

    return df

def delDuplicates(train_dataset):
    #print the number of rows
    print(train_dataset.shape[0]) 
    #delete any duplicates
    dataset = train_dataset.drop_duplicates()
    print(train_dataset.shape[0])
        
def findUniqueVals(df):
    #columns = df.columns    
    #for column in columns:
        #print(column)
        #print(df[column].unique())
        #print(len(df[column].unique()))
        
        #Year of Record: nan
        #Housing Situation: 0 '0' 'nA' <-object
        #Crime Level in the City of Employement
        #Work Experience in Current Job [years]: '#NUM!' 23.0 '23' ... <-object
        #Satisfation with employer: nan <-object
        #Gender: 'other' nan 'unknown' '0' 'f' 'female' <-object
        #Age
        #Country <-object
        #Size of City
        #Profession <-object (1355 unique values)
        #University Degree: 'No' nan '0' <-object
        #Hair Color: nan 'Unknown' '0' <-object
        #Body Height [cm]
        #Yearly Income in addition to Salary (e.g. Rental Income): '0 EUR' <-object
        #Total Yearly Income [EUR] => y
    
    df['Gender'] = df['Gender'].replace('f', 'female')
    df['Gender'] = df['Gender'].replace('0', np.nan)
    df['Gender'] = df['Gender'].replace('unknown', np.nan)
    
    df['University Degree'] = df['University Degree'].replace('0', np.nan)
    
    df['Work Experience in Current Job [years]'] = df['Work Experience in Current Job [years]'].replace('#NUM!', np.nan)
    df['Work Experience in Current Job [years]'] = df['Work Experience in Current Job [years]'].astype(float)
    
    df['Housing Situation'] = df['Housing Situation'].replace([0, '0', 'nA'], np.nan)
    
    df['Yearly Income in addition to Salary (e.g. Rental Income)'] = df['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x: x.lstrip('+-').rstrip('EUR'))
    df['Yearly Income in addition to Salary (e.g. Rental Income)'] = df['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)

    return df

if __name__ == '__main__':
    main()
