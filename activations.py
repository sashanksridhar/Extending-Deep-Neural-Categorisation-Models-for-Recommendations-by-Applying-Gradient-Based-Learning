# import the necessary packages
from GradCAM import GradCAMS

from scipy import sparse
import numpy as np
import time
from sklearn.preprocessing import scale
import csv
# construct the argument parser and parse the arguments

from tensorflow.keras.models import load_model
model=load_model("epoch300.h5")
print(model.summary())


Xi = []
movies = []

with open("input.csv", 'r') as r:
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c == 0:
            c += 1
            continue
        op = []
        for j in range(0, len(row)):
            if j==24:
                # print(row[j])
                movies.append(row[j])
                continue
            op.append(float(row[j]))
        Xi.append(op)
        c+=1

print(c)

X = []
Y = []
for i in Xi:
    X.append(i[:len(i)-5])
    Y.append(i[len(i)-5:])

print(len(X))



for i in range(77851,len(X)):
    if(i%100==0):
        print(i)

    # print(np.array(X[i]).shape)

    preds = model.predict([X[i],])


    ind = np.argmax(preds)
    # print(ind)
    cam = GradCAMS(model, ind)

    val1 = cam.compute_heatmap(X[i])



    a_sparse = sparse.csr_matrix(val1)

    a1 =[]
    a1.append(movies[i])

    # print(a_sparse)



    for i in range(32):
        a1.append(a_sparse[0, i])
    # break
    with open("activations.csv", 'a', encoding='latin1') as csvWrite:
        filewriter = csv.writer(csvWrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

        filewriter.writerow(a1)









