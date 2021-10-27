import random, os, cv2, pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.datasets import make_classification

dir = "F:/Surf Road Detect/TRAIN&TEST/trainImages" # change here

# categories = ['Pothole', 'Road Marking']

categories = ['Alligator', 'Joint', 'Longitudal', 'Manholes', 'Oil Marks', 'Pothole', 'Road Marking', 'Shadow',
              'Transverse']


data = []
pca = PCA(.95)
# **img pre-processing
for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        road_img = cv2.imread(img_path, 0)
        resized = cv2.resize(road_img, (32,32), interpolation=cv2.INTER_AREA)
        road_img = resized
        try:
            image = np.array(road_img)
            ##feature extraction
            image = StandardScaler().fit_transform(image)
            image = pca.fit_transform(image)
            image = image.flatten()
            print(len((image)))
            # image = np.array(road_img)
            data.append(np.array([image, label]))

        except Exception as e:
            pass

print('data len', len(data))

# # write  the pickle data file: pick_in
# pick_in = open('data1.pickle', 'wb')
# pickle.dump(data, pick_in)
# pick_in.close()
#
# print('data dump complete')
#
# # read the pickle file: pick_in
# pick_in = open('data1.pickle', 'rb')
# data = pickle.load(pick_in)
# pick_in.close()
#
# print('data read complete')


# random.shuffle(data)
#
# print('shuffle complete')
#
# features = []
# labels = []
#
# for feature, label in data:
#     features.append(feature)
#     labels.append(label)
#
# print('feature and label load complete')
#
# x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
#
# print('split complete')
# df1 = pd.DataFrame(x_train)
# df2 = pd.DataFrame(y_train)
# print(df1.shape, df2.shape)

###########################################################

# model = LinearSVC(random_state=0, tol=1e-5, verbose=True)
# model.fit(x_train, y_train)
# print('lsvc fit complete')
#
# prediction = model.predict(x_test)
# accuracy = model.score(x_test, y_test)
# print('lsvc Accuracy:', accuracy)

###########################################################
#
# model = SGDClassifier(verbose=True)
# model.fit(x_train, y_train)
# print('sgd fit complete')
#
# prediction = model.predict(x_test)
# accuracy = model.score(x_test, y_test)
# print('sgd Accuracy:', accuracy)

###########################################################

# model = neighbors.KNeighborsClassifier()
# model.fit(x_train, y_train)
# print('KNN fit complete')
#
# prediction = model.predict(x_test)
# accuracy = model.score(x_test, y_test)
# print('KNN Accuracy:', accuracy)

###########################################################
#tunning
# param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"], "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}
#
# base_estimator = LogisticRegression(random_state=0)
# X, y = make_classification(n_samples=1000, random_state=0)
# best_clf = GridSearchCV(base_estimator, param_grid, cv=5, n_jobs=-1).fit(X, y)
# print(best_clf.best_estimator_)
# model = LogisticRegression(C=0.1, penalty='l1', random_state=0, solver='saga')
#
# model.fit(x_train, y_train)
# print('Log fit complete')
#
# prediction = model.predict(x_test)
# accuracy = model.score(x_test, y_test)
# print('Log Accuracy:', accuracy)

###########################################################
#tunning

# param_grid = {'max_depth': [3, 5, 10],
#              'min_samples_split': [2, 5, 10]}
# base_estimator = RandomForestClassifier(random_state=0)
# X, y = make_classification(n_samples=1000, random_state=0)
# sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
#                             factor=2, resource='n_estimators',
#                                 max_resources=30).fit(X, y)
# print(sh.best_estimator_)
# model = RandomForestClassifier(max_depth=5, n_estimators=24, random_state=0)
# model.fit(x_train, y_train)
# print('RF fit complete')
#
# prediction = model.predict(x_test)
# accuracy = model.score(x_test, y_test)
# print('RF Accuracy:', accuracy)

###########################################################

# model = GaussianNB(verbose=True) # Continuous variable
# model.fit(x_train, y_train)
# print('NB fit complete')
#
# prediction = model.predict(x_test)
# accuracy = model.score(x_test, y_test)
# print('NB Accuracy:', accuracy)


###########################################################

# model = tree.DecisionTreeClassifier()
# model.fit(x_train, y_train)
# print('DT fit complete')
#
# prediction = model.predict(x_test)
# accuracy = model.score(x_test, y_test)
# print('DT Accuracy:', accuracy)

###########################################################

# model = GradientBoostingClassifier(verbose=True)
# model.fit(x_train, y_train)
# print('GB fit complete')
#
# prediction = model.predict(x_test)
# accuracy = model.score(x_test, y_test)
# print('GB Accuracy:', accuracy)

###########################################################
## Confusion Mat
###########################################################
# print("===== Confusion Matrix ======")
# from sklearn.metrics import classification_report, confusion_matrix
# print("Consusion Matrix:\n%s" % confusion_matrix(y_test, prediction))
# print("Classification Report:\n%s" % classification_report(y_test, prediction))