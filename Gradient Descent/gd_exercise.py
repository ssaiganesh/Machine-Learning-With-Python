import pandas as pd
import numpy as np
import math 

url = 'https://raw.githubusercontent.com/codebasics/py/master/ML/3_gradient_descent/Exercise/test_scores.csv'
df = pd.read_csv(url, error_bad_lines=False)

def gradient_descent(x,y):
    m_curr = b_curr = 0 
    iterations = 1000000 #had to refer to answer 
    n = len(x) 
    learning_rate = 0.0002 
    cost_previous = 0
    for i in range(iterations):
        #To see the equations please refer to gradient descent notes file
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n) * sum(x*(y-y_predicted)) 
        bd = -(2/n) * sum(y - y_predicted) 
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print(f'm {m_curr}, b {b_curr}, cost {cost} iteration {i}') 
        #it can be seen that the cost is reducing


x = np.array(df.math)
y = np.array(df.cs)

gradient_descent(x,y)