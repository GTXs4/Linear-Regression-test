from methods import linear_regression
from sklearn.linear_model import LinearRegression 
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from tabulate import tabulate

#TEST DATA
np.random.seed(12)
n=13
data = pd.DataFrame({"x1":5.1 + 5.0*np.random.randn(n),
                      "x2":1.3 + 3.2*np.random.randn(n),
                      "x3":3.7 + 1.3*np.random.randn(n),
                      "y":25.2 + 12.3*np.random.randn(n)})

x1 = data["x1"].tolist()
x2 = data["x2"].tolist()
x3 = data["x3"].tolist()
y = data["y"].tolist()

if __name__ == "__main__":
    #df.head()
    intercepts = []
    coef_x1 = []
    coef_x2 = []
    coef_x3 = []
    R2s = []
    predictions = []
    x_pred = [2.5, 7.20, 9.12]
    X_pred = pd.DataFrame({"x1":[x_pred[0]],"x2":[x_pred[1]],"x3":[x_pred[2]]})
    
    #OWN IMPLEMENTATION
    lm1 = linear_regression(x1,x2,x3,y=y)
    print("Implementation Summary\n")
    lm1.summary()
        
    intercepts.append(round(lm1.coef.item(0),5))
    coef_x1.append(round(lm1.coef.item(1),5))
    coef_x2.append(round(lm1.coef.item(2),5))
    coef_x3.append(round(lm1.coef.item(3),5))
    R2s.append(round(lm1.R2,5))
    predictions.append(round(lm1.prediction_(x_pred),5))
          
    X = data[["x1","x2","x3"]]
    Y = data["y"]
    
    #SCIKIT LEARN
    lm2 = LinearRegression().fit(X,Y)
    
    intercepts.append(round(lm2.intercept_,5))
    coef_x1.append(round(lm2.coef_[0],5))
    coef_x2.append(round(lm2.coef_[1],5))
    coef_x3.append(round(lm2.coef_[2],5))
    R2s.append(round(lm2.score(X,Y),5))
    predictions.append(round(lm2.predict(X_pred).item(0),5))
    
    #STATSMODELS
    lm3 = smf.ols(formula = "y ~ x1 + x2 + x3", data = data).fit()
    intercepts.append(round(lm3.params[0],5))
    coef_x1.append(round(lm3.params[1],5))
    coef_x2.append(round(lm3.params[2],5))
    coef_x3.append(round(lm3.params[3],5))
    R2s.append(round(lm3.rsquared,5))
    predictions.append(round(lm3.predict(X_pred),5))
    
    #print(lm3.rsquared_adj)
    print("\nComparison of the results of the models generated with different implementations \n")
    names = ["implementation","scikit","statmodels"]
    table = []
    for i in range(len(names)):
        table.append([names[i], intercepts[i], coef_x1[i], coef_x2[i], coef_x3[i], R2s[i], predictions[i]])
    print(tabulate(table, headers=["Model","intercept","coef_x1","coef_x2","coef_x3","R2","prediction"]))
    
    
