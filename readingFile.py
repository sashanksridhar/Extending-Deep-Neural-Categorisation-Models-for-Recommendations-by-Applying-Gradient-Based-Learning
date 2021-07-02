import numpy as np
import pandas as pd
from sklearn import preprocessing
import csv

column_names = ['user_id', 'item_id', 'rating','timestamp']
df = pd.read_csv('E:\\NewsRecProposal\\archive\\ml-100k\\u.data', sep='\t', names=column_names)

FieldsMovies = ['movieID', 'movieTitle', 'releaseDate', 'videoReleaseDate', 'IMDbURL', 'unknown', 'action', 'adventure',
      'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'filmNoir', 'horror',
      'musical', 'mystery', 'romance','sciFi', 'thriller', 'war', 'western']

Itemdata = pd.read_csv('E:\\NewsRecProposal\\archive\\ml-100k\\u.item', sep='|', encoding = "ISO-8859-1", names=FieldsMovies)

UserFields = ['user id','age','gender','occupation','zipcode']

userData = pd.read_csv('E:\\NewsRecProposal\\archive\\ml-100k\\u.user', sep='|', names=UserFields)

Itemdata = Itemdata.drop(['releaseDate','videoReleaseDate', 'IMDbURL', 'unknown'], axis=1)

df = df.drop('timestamp', axis=1)

userData = pd.concat([userData, pd.get_dummies(userData.gender, prefix='Gender')], axis=1)

userData = pd.concat([userData, pd.get_dummies(userData.occupation, prefix='Job')], axis=1)

userData = userData.drop(['zipcode','gender','occupation'], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
x = userData['age'].values
x = x.reshape(-1,1)
x_scaled = min_max_scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=['age'], index = userData.index)
userData['age'] = df_temp

print(userData.head())
print(userData.keys())

df = pd.concat([df, pd.get_dummies(df.rating, prefix='rating')], axis=1)
df = df.drop(['rating'], axis=1)
print(df.head())
print(df.keys())

print(Itemdata.keys())

with open("input.csv", 'a', encoding='latin1') as csv_file:
    filewriter = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    headings = ['age', 'Gender_F', 'Gender_M', 'Job_administrator',
       'Job_artist', 'Job_doctor', 'Job_educator', 'Job_engineer',
       'Job_entertainment', 'Job_executive', 'Job_healthcare', 'Job_homemaker',
       'Job_lawyer', 'Job_librarian', 'Job_marketing', 'Job_none', 'Job_other',
       'Job_programmer', 'Job_retired', 'Job_salesman', 'Job_scientist',
       'Job_student', 'Job_technician', 'Job_writer', 'movieTitle', 'action', 'adventure', 'animation',
       'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy',
       'filmNoir', 'horror', 'musical', 'mystery', 'romance', 'sciFi',
       'thriller', 'war', 'western', 'rating_1', 'rating_2', 'rating_3', 'rating_4',
       'rating_5']
    filewriter.writerow(headings)

    for ind in df.index:
        if ind%100==0:
            print(ind)
        data=[]
        userid = df['user_id'][ind]
        itemid = df['item_id'][ind]
        userrow = userData.loc[userData['user id']==userid]
        data.append(userrow['age'].values[0])
        data.append(userrow['Gender_F'].values[0])
        data.append(userrow['Gender_M'].values[0])
        data.append(userrow['Job_administrator'].values[0])
        data.append(userrow['Job_artist'].values[0])
        data.append(userrow['Job_doctor'].values[0])
        data.append(userrow['Job_educator'].values[0])
        data.append(userrow['Job_engineer'].values[0])
        data.append(userrow['Job_entertainment'].values[0])
        data.append(userrow['Job_executive'].values[0])
        data.append(userrow['Job_healthcare'].values[0])
        data.append(userrow['Job_homemaker'].values[0])
        data.append(userrow['Job_lawyer'].values[0])
        data.append(userrow['Job_librarian'].values[0])
        data.append(userrow['Job_marketing'].values[0])
        data.append(userrow['Job_none'].values[0])
        data.append(userrow['Job_other'].values[0])
        data.append(userrow['Job_programmer'].values[0])
        data.append(userrow['Job_retired'].values[0])
        data.append(userrow['Job_salesman'].values[0])
        data.append(userrow['Job_scientist'].values[0])
        data.append(userrow['Job_student'].values[0])
        data.append(userrow['Job_technician'].values[0])
        data.append(userrow['Job_writer'].values[0])

        itemrow = Itemdata.loc[Itemdata['movieID'] == itemid]
        data.append(itemrow['movieTitle'].values[0])
        data.append(itemrow['action'].values[0])
        data.append(itemrow['adventure'].values[0])
        data.append(itemrow['animation'].values[0])
        data.append(itemrow['childrens'].values[0])
        data.append(itemrow['comedy'].values[0])
        data.append(itemrow['crime'].values[0])
        data.append(itemrow['documentary'].values[0])
        data.append(itemrow['drama'].values[0])
        data.append(itemrow['fantasy'].values[0])
        data.append(itemrow['filmNoir'].values[0])
        data.append(itemrow['horror'].values[0])
        data.append(itemrow['musical'].values[0])
        data.append(itemrow['mystery'].values[0])
        data.append(itemrow['romance'].values[0])
        data.append(itemrow['sciFi'].values[0])
        data.append(itemrow['thriller'].values[0])
        data.append(itemrow['war'].values[0])
        data.append(itemrow['western'].values[0])

        data.append(df['rating_1'][ind])
        data.append(df['rating_2'][ind])
        data.append(df['rating_3'][ind])
        data.append(df['rating_4'][ind])
        data.append(df['rating_5'][ind])

        filewriter.writerow(data)

# print(df['user_id'][10000])
# for i in range(0,df.size):
     # print(df.iloc[i])
