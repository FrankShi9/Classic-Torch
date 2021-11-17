from sklearn import datasets
from sklearn.model_selection import train_test_split
import pickle, random


# read the pickle file: pick_in
pick_in = open('data1.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

print('data read complete')


random.shuffle(data)

print('shuffle complete')

features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

print('feature and label load complete')

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


import numpy as np
import math
from collections import Counter

# def tDW(h, y):
Q_dp = [
        [

        ]
      ] #DP 2-D list

q_dp = [
        [

        ]
      ] #DP 2-D list
# border case relaxed
def P2DHMDM(h, y): #!! input img data array must not be flattened !!
    buf = 0.
    for i in h[:, 0]:
        for x in y[:, 0]:
            for j in h[0, :]:
                for y in y[0, :]:
                        if x >= 1 and j >=1 and y >= 2:
                            Q_ijxy = math.sqrt((math.fabs((h[i, j]) - min(y[x-1,y], y[x, y], y[x+1, y])))**2) + min(Q_dp[j-1, y-2], Q_dp[j-1, y-1], Q_dp[j-1, y])
                            Q_dp.append(Q_ijxy) #change insert location
                        else:
                            Q_ijxy = math.sqrt((math.fabs(h[i,j] - y[x, y]))**2)
                            Q_dp.append(Q_ijxy)

            q_ij = Q_dp[0,:] + min(q_dp[j-1, y-2], q_dp[j-1, y-1], q_dp[j-1, y])

    return q_dp[:,:].sum()

# Implement KNN
def kNNClassify(K, X_train, y_train, X_predict): # LAZY classifier
    # distances = [sqrt(np.sum((x-X_predict)**2)) for x in X_train]  # change distance measure here
    distances = [P2DHMDM(X_predict, x) for x in X_train]
    sort = np.argsort(distances) # Index list
    topK = [y_train[i] for i in sort[:K]]
    votes = Counter(topK)
    y_predict = votes.most_common(1)[0][0]
    return y_predict

def kNN_predict(K, X_train, y_train, X_predict, y_predict):  # accuracy counter
    correct = 0
    for i in range(len(X_predict)):
        if y_predict[i] == kNNClassify(K, X_train, y_train, X_predict[i]):
            correct += 1

    print(correct/len(X_predict))


print("Training accuracy is ", end='')
kNN_predict(3, X_train, y_train, X_train, y_train)
print("Test accuracy is ", end='')
kNN_predict(3, X_train, y_train, X_test, y_test)

