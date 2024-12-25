from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC

df=pd.read_csv('preprocessed_data1.csv')
df.dropna(subset=['tweet'], inplace=True)
vectorizer = TfidfVectorizer(max_features=1000)

X_text = vectorizer.fit_transform(df['tweet'])

X_numerical = df[['User ID', 'Retweet Count']].values
X_combined = np.hstack((X_numerical, X_text.toarray()))
Y = df['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

pickle.dump(rf_classifier, open('m.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

nb_classifier = GaussianNB()  
nb_classifier.fit(X_train, y_train)
pickle.dump(nb_classifier, open('nb.pkl', 'wb'))

svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)
pickle.dump(svm_classifier, open('svm.pkl', 'wb'))

logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
logistic_regression.fit(X_train, y_train)
pickle.dump(logistic_regression, open('lr.pkl', 'wb'))

from sklearn.tree import DecisionTreeClassifier

# Initialize and train the Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train, y_train)
pickle.dump(logistic_regression, open('dt.pkl', 'wb'))

from sklearn.neighbors import KNeighborsClassifier

# Initialize and train the KNN classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
pickle.dump(logistic_regression, open('knn.pkl', 'wb'))

from sklearn.ensemble import GradientBoostingClassifier

# Initialize and train the GBM classifier
gbm_classifier = GradientBoostingClassifier(random_state=42)
gbm_classifier.fit(X_train, y_train)
pickle.dump(logistic_regression, open('gbm.pkl', 'wb'))

from sklearn.ensemble import AdaBoostClassifier

# Initialize and train the AdaBoost classifier
adaboost_classifier = AdaBoostClassifier(random_state=42)
adaboost_classifier.fit(X_train, y_train)
pickle.dump(logistic_regression, open('ab.pkl', 'wb'))

from sklearn.neural_network import MLPClassifier

# Initialize and train the MLP classifier
mlp_classifier = MLPClassifier(random_state=42)
mlp_classifier.fit(X_train, y_train)
pickle.dump(logistic_regression, open('mlp.pkl', 'wb'))
