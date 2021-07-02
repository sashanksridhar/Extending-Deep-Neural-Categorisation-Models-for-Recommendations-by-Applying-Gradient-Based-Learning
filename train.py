import csv

import numpy
from keras import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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

train ,test = train_test_split(Xi,test_size=0.25)
X = []
Xt = []
Yt = []
Y = []
for i in train:
    X.append(i[:len(i)-5])
    Y.append(i[len(i)-5:])
for i in test:
    Xt.append(i[:len(i)-5])
    Yt.append(i[len(i) - 5:])

model = Sequential()
model.add(Dense(64, input_dim=42, activation='relu'))
model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
# model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(5, activation='sigmoid'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

Xout = numpy.array(X)
Y = numpy.array(Y)
model.fit(Xout, Y, epochs=300,batch_size=256)

Xtest = numpy.array(Xt)
Yt = numpy.array(Yt)
print(model.evaluate(Xtest,Yt))

predicted_labels = model.predict(Xtest)

predicted_labels = numpy.array((predicted_labels>0.5))

cnf_matrix = confusion_matrix(Yt.argmax(axis=1), predicted_labels.argmax(axis=1))
print(cnf_matrix)

FP = cnf_matrix.sum(axis=0) - numpy.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - numpy.diag(cnf_matrix)
TP = numpy.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)

print(TP)
print(TN)
print(FP)
print(FN)
print(TPR)
print(TNR)
print(FPR)
print(FNR)
print(PPV)
print(NPV)
print(FDR)
print(ACC)

model.save("epoch300.h5")