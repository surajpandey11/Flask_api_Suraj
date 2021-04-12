

import numpy as np
import pandas as pd
from sklearn import datasets
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
def model_predict():
    diabetes = datasets.load_diabetes()
    X=pd.DataFrame(data=diabetes.data,columns=diabetes.feature_names)

    
    
    y=(diabetes.target)
    # print(X)
    # print(y)
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    
    model=LinearRegression()
    model.fit(X_train,y_train)

    # a= [[0.038076 , 0.050680 , 0.061696 , 0.021872, -0.044223 ,-0.034821, -0.043401, -0.002592,  0.019908, -0.017646]]
    # print(model.predict(a))
    pickle.dump(model, open('models/diabetes_final.pickle', 'wb'))
    
model_predict()

