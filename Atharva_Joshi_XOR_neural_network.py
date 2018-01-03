import numpy as np

#training data
x = np.array([ [0,1],[1,0],[1,1],[0,0] ])
y = np.array([[1,1,0,0]]).T

#generating random weights with mean zero for 2 layers of neural network
first_weights = 2*np.random.random((2,4)) - 1
second_weights = 2*np.random.random((4,1)) - 1

#training neural network
for p in range(100000):
    #using sigmoid as an output of one neuron
    out_1 = 1/(1+np.exp(-(np.dot(x,first_weights))))
    #using sigmoid as an output of second neuron
    out_2 = 1/(1+np.exp(-(np.dot(out_1,second_weights))))
    #calculating loss and slope
    change_2 = (y - out_2)*(out_2*(1-out_2))
    change_1 = change_2.dot(second_weights.T) * (out_1 * (1-out_1))
    #updating weights
    second_weights += out_1.T.dot(change_2)
    first_weights += x.T.dot(change_1)

#predicting using obtained weights from trained neural net
    
def predict(t):
    e_1 = 1/(1+np.exp(-(np.dot(t,first_weights))))
    e_2 = 1/(1+np.exp(-(np.dot(e_1,second_weights))))
    return e_2
#function to take input and produce output in desired format
"""to predict type predictnow('x','y'),type x and y strictly in single quotation marks"""
def predictnow(x,y):
    x=str(x)
    y=str(y)
    a=[int(i) for i in list(x)]
    a_1=a[0]
    a_2=a[1]
    b=[int(i) for i in list(y)]
    b_1=b[0]
    b_2=b[1]
    a=[a_1,b_1]
    b=[a_2,b_2]
    a=np.asarray(a)
    b=np.asarray(b)
    c=predict(a)
    d=predict(b)
    c=c[0]
    d=d[0]
    c = int(round(c))
    d = int(round(d))
    c=str(c) 
    d=str(d)
    f=c+d
    return f

predictnow('11','11')

