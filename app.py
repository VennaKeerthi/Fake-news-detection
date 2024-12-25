from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model1=pickle.load(open('m.pkl', 'rb'))
model2=pickle.load(open('nb.pkl', 'rb'))
model3=pickle.load(open('svm.pkl', 'rb'))
model4=pickle.load(open('dt.pkl', 'rb'))
model5=pickle.load(open('knn.pkl', 'rb'))
model6=pickle.load(open('gbm.pkl', 'rb'))
model7=pickle.load(open('ab.pkl', 'rb'))
model8=pickle.load(open('mlp.pkl', 'rb'))
model9=pickle.load(open('lr.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app=Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = int(request.form['a'])
    data2 = int(request.form['b'])
    data3 = request.form['c']
    
    data3_encoded = vectorizer.transform([data3])

    arr = np.hstack((np.array([[data1, data2]]), data3_encoded.toarray()))
    model_name = request.form['d']
    if model_name == 'rf':
        pred = model1.predict(arr)
        return render_template('after.html', data=pred, model='Random Forest')
    
    elif model_name == 'nb':
        pred = model2.predict(arr)
        return render_template('after.html', data=pred, model='Naive Bayes')
    
    elif model_name == 'svm':
        pred = model3.predict(arr)
        return render_template('after.html', data=pred, model='Support Vector Machine')
    
    elif model_name == 'dt':
        pred = model4.predict(arr)
        return render_template('after.html', data=pred, model='Decision Tree')
    
    elif model_name == 'knn':
        pred = model5.predict(arr)
        return render_template('after.html', data=pred, model='K-Nearest Neighbour')
    
    elif model_name == 'gbm':
        pred = model6.predict(arr)
        return render_template('after.html', data=pred, model='Gradient Boosting Machine')
    
    elif model_name == 'ab':
        pred = model7.predict(arr)
        return render_template('after.html', data=pred, model='AdaBoost')
    
    elif model_name == 'mlp':
        pred = model8.predict(arr)
        return render_template('after.html', data=pred, model='Multi Layer Perceptron')
    
    elif model_name == 'lr':
        pred = model9.predict(arr)
        return render_template('after.html', data=pred, model='Logistic Regression')
    
    else:
        pred = "Invalid model selection"
        return render_template('after.html', data=pred)
    

if __name__=="__main__":
    app.run(debug=True)

