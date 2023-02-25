import pandas as pd
import os
import re
import numpy as np

data = pd.read_csv('./data/full.csv')
data.head()

# regex to extract feet and inches from height
regex_feet = re.compile("([0-9]+)\'")
regex_inches = re.compile("([0-9]+)\"")

def extract_numbers(x, regex):
    captures = regex.findall(x)
    if len(captures) > 0:
        try:
            return int(captures[0])
        except:
            return None
    else:
        return None
    
# parse the height string to feet and inches
data['feet'] = data['height'].map(lambda i: extract_numbers(i, regex_feet))
data['inches'] = data['height'].map(lambda i: extract_numbers(i, regex_inches))

# found some incorrect inches, assume 63" to 6.3"
data['inches'] = data['inches'].map(lambda i: i / 10 if i > 12 else i)
# convert feet/inches to inches
data['height'] = data.apply(lambda row: row['feet'] * 12 + row['inches'], axis=1)
# covert inches to m
data['height'] = data['height'].map(lambda i: i * 2.54 / 100)
# weight pounds => kg
data['weight'] = data['weight'].map(lambda i: i * 0.453592)
# calculate BMI = weight/height^2
data['bmi'] = data.apply(lambda row: row['weight'] / row['height'] / row['height'], axis = 1)
# create gender (number format of sex)
data['gender'] = data['sex'].map(lambda i: 1 if i == 'Male' else 0)

data[['nameid','age','height','weight','race','sex','eyes','hair', 'bmi']].head()

# remove rows that no face images
data['index'] = data['bookid'].map(lambda i: str(i) +'.jpg')
allimage = os.listdir('./data/face/')
data = data.loc[data['index'].isin(allimage),:]

# remove rows with invalid BMI
data = data.loc[~data['bmi'].isnull(), :]

# split train/valid
in_train = np.random.random(size = len(data)) <= 0.8
train = data.loc[in_train,:]
test = data.loc[~in_train,:]

print('train data dimension: {}'.format(str(train.shape)))
print('test data dimension:  {}'.format(str(test.shape)))

# output to csv files
train.to_csv('./data/train.csv', index = False)
test.to_csv('./data/valid.csv', index = False)



