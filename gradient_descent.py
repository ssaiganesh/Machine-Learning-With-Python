import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0 
    iterations = 10000
    n = len(x) #assuming x and y len is same
    learning_rate = 0.08 #trial and error

    for i in range(iterations):
        #To see the equations please refer to gradient descent notes file
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n) * sum(x*(y-y_predicted)) 
        bd = -(2/n) * sum(y - y_predicted) 
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print(f'm {m_curr}, b {b_curr}, cost {cost} iteration {i}') 
        #it can be seen that the cost is reducing

         
#using numpy arrays for convenient matrix multiplication and faster compared to python lists
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

#we can see from simple math the above equation is actually y  = 2x+3 we need to reach that values in the gradient descent 

gradient_descent(x,y)

    
    #always start with lesser iterations and check if cost is reducing . 
    #if want to take bigger steps in reducing, use larger learning rate. 
    #if still not possible increase iterations to reach zero. in this case reaches 1 x 10 ^ -29 . 
    #It can also be seen that cost remains about the same once closer to minimum 
    
