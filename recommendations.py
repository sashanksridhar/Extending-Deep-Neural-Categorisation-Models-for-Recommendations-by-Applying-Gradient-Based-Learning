import pandas as pd
import numpy as np
import csv
import math
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
from scipy import sparse

from sklearn import preprocessing

# df = pd.read_csv("activations.csv",encoding='latin1')

Xi = []
movies = []

with open("activations.csv", 'r', encoding='latin1') as r:
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c == 0:
            c += 1
            continue
        op = []
        for j in range(0, len(row)):
            if j == 0:
                # print(row[j])
                movies.append(row[j])
                continue
            op.append(float(row[j]))
        Xi.append(op)
        c += 1

genre = []

dict_genre = {25: "Action", 26: "Adventure", 27: "Animation",
              28: "Children's", 29: "Comedy", 30: "Crime", 31: "Documentary", 32: "Drama", 33: "Fantasy",
              34: "Film-Noir", 35: "Horror", 36: "Musical", 37: "Mystery", 38: "Romance", 39: "Sci-Fi",
              40: "Thriller", 41: "War", 42: "Western"}

with open("input.csv", 'r', encoding='latin1') as r:
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c == 0:
            c += 1
            continue
        op = []
        for j in range(0, len(row)):
            if j in range(25, 42) and int(row[j] == 1):
                # print(row[j])
                genre.append(dict_genre[j])
                continue

        c += 1

Xinput = []
movies = []

with open("input.csv", 'r', encoding='latin1') as r:
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c == 0:
            c += 1
            continue
        op = []
        for j in range(0, len(row) - 5):
            if j == 24:
                # print(row[j])
                movies.append(row[j])
                continue
            op.append(float(row[j]))
        Xinput.append(op)
        c += 1

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('E:\\NewsRecProposal\\archive\\ml-100k\\u.data', sep='\t', names=column_names)

FieldsMovies = ['movieID', 'movieTitle', 'releaseDate', 'videoReleaseDate', 'IMDbURL', 'unknown', 'action', 'adventure',
                'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'filmNoir', 'horror',
                'musical', 'mystery', 'romance', 'sciFi', 'thriller', 'war', 'western']

Itemdata = pd.read_csv('E:\\NewsRecProposal\\archive\\ml-100k\\u.item', sep='|', encoding="ISO-8859-1", names=FieldsMovies)

UserFields = ['user id', 'age', 'gender', 'occupation', 'zipcode']

userData = pd.read_csv('E:\\NewsRecProposal\\archive\\ml-100k\\u.user', sep='|', names=UserFields)

Itemdata = Itemdata.drop(['releaseDate', 'videoReleaseDate', 'IMDbURL', 'unknown'], axis=1)

df = df.drop('timestamp', axis=1)

userData = pd.concat([userData, pd.get_dummies(userData.gender, prefix='Gender')], axis=1)

userData = pd.concat([userData, pd.get_dummies(userData.occupation, prefix='Job')], axis=1)

userData = userData.drop(['zipcode', 'gender', 'occupation'], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
x = userData['age'].values
x = x.reshape(-1, 1)
x_scaled = min_max_scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=['age'], index=userData.index)
userData['age'] = df_temp

print(userData.head())
print(userData.keys())

df = pd.concat([df, pd.get_dummies(df.rating, prefix='rating')], axis=1)
df = df.drop(['rating'], axis=1)
print(df.head())
print(df.keys())

print(Itemdata.keys())

from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')

# XK = Xi
# YK = movies

model_knn.fit(Xi, movies)

from tensorflow.keras.models import load_model
from GradCAM import GradCAMS
import keras

model = load_model("epoch300.h5")

recall = 0
recallden = 0
countnum = 0
precision = 0
precisionden = 0
wscore = 0

ratingdict = {}
index = 10

preds = model.predict([Xinput[index], ])
keras.backend.clear_session()

ind = np.argmax(preds)
cam = GradCAMS(model, ind)

val1 = cam.compute_heatmap(Xinput[index])

a_sparse = sparse.csr_matrix(val1)

a1 = []

for i in range(32):
    a1.append(a_sparse[0, i])

    # print(movies[index])

    # a1 = Xi[index]

distances, indices = model_knn.kneighbors(np.array(a1).reshape(1, -1), n_neighbors=50)  # change neighbors 25, 50

userid = df['user_id'][index]
# # print(userid)
userList = df[df['user_id'] == userid].index.tolist()

# print(userList)
#

Fields_movies = ['action', 'adventure',
                'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'filmNoir', 'horror',
                'musical', 'mystery', 'romance', 'sciFi', 'thriller', 'war', 'western']
genre_count = {}

for f in Fields_movies:
    genre_count[f] = 0
for j in userList:
  x = df.loc[[j]]
  # print(x)
  # print(x['rating_1'][0])
  rating = -1
  if x['rating_1'].iloc[0]==1:
    rating=1
  elif x['rating_2'].iloc[0]==1:
    rating=2
  elif x['rating_3'].iloc[0]==1:
    rating=3
    for field in Fields_movies:
        if Itemdata.loc[[x['item_id'].iloc[0]]][field].iloc[0] == 1:
            genre_count[field]+=1
    # print(movies[x['item_id'].iloc[0]])
  elif x['rating_4'].iloc[0]==1:
    rating=4
    # print(movies[x['item_id'].iloc[0]])
    # print(Itemdata.loc[[x['item_id'].iloc[0]]]["movieTitle"].iloc[0])
    for field in Fields_movies:
        if Itemdata.loc[[x['item_id'].iloc[0]]][field].iloc[0] == 1:
            genre_count[field]+=1
  elif x['rating_5'].iloc[0]==1:
    rating=5
    for field in Fields_movies:
        if Itemdata.loc[[x['item_id'].iloc[0]]][field].iloc[0] == 1:
            genre_count[field]+=1
    # print(movies[x['item_id'].iloc[0]])

print(genre_count)

# for j in range(0,len(indices[0])):
#     indices[0][j]+=1


# for j in userList:
#     Iu+=1

#     # itemid = df._get_value(j,'item_id')
#     # print(itemid)
#     if j in indices[0]:
#         wi+=1

print("******")
print("predictions")
print("******")

genre_count = {}

for f in Fields_movies:
    genre_count[f] = 0

for j in indices[0]:

    datafield = df.loc[[j]]
    preds = model.predict([Xinput[j], ])

    ind = np.argmax(preds)
    # print(ind)
    if ind==3:
        # print(print(Itemdata.loc[[datafield['item_id'].iloc[0]]]["movieTitle"].iloc[0]))
        # print(Itemdata.loc[[datafield['item_id'].iloc[0]]])
        for field in Fields_movies:
            if Itemdata.loc[[datafield['item_id'].iloc[0]]][field].iloc[0]==1:
                # print(field, sep="\t")
                genre_count[field] += 1
    if ind==4:
        print(print(Itemdata.loc[[datafield['item_id'].iloc[0]]]["movieTitle"].iloc[0]))
        # print(Itemdata.loc[[datafield['item_id'].iloc[0]]])
        for field in Fields_movies:
            if Itemdata.loc[[datafield['item_id'].iloc[0]]][field].iloc[0]==1:
                print(field, sep="\t")
                genre_count[field] += 1
    if ind==5:
        # print(print(Itemdata.loc[[datafield['item_id'].iloc[0]]]["movieTitle"].iloc[0]))
        # print(Itemdata.loc[[datafield['item_id'].iloc[0]]])
        for field in Fields_movies:
            if Itemdata.loc[[datafield['item_id'].iloc[0]]][field].iloc[0]==1:
                # print(field, sep="\t")
                genre_count[field] += 1


print(genre_count)

    # print(datafield)
    # rating = -1
    # if datafield['rating_1'].iloc[0] == 1:
    #     rating = 1
    # elif datafield['rating_2'].iloc[0] == 1:
    #     rating = 2
    # elif datafield['rating_3'].iloc[0] == 1:
    #     rating = 3
    # elif datafield['rating_4'].iloc[0] == 1:
    #     rating = 4
    # elif datafield['rating_5'].iloc[0] == 1:
    #     rating = 5

# if rating >= 3:
#     recommend += 1
# elif rating < 3 and rating != -1:
#     notrecommend += 1
#
#     ind = -1
#
#     if ratingdict.get(j) == None:
#         preds = model.predict([Xinput[j], ])
#
#         ind = np.argmax(preds)
#
#         ratingdict[j] = ind
#
#     else:
#         ind = ratingdict.get(j)
#
#     if ind >= 3:
#         relevant += 1
#     elif ind < 3 and ind != -1:
#         irrelevant += 1
#
#     if ind >= 3 and rating >= 3:
#         prec += 1


