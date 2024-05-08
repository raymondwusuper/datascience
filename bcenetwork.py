#Neural Network
import numpy as np
def sigFunc(x):
    #activation function, other activation functions include tanh, linear, sigmoid
    #I will use sigmoid because it is the easiest and most simple (and I know how to use it)
    return 1/(1+np.exp(-x))

#derivative of the activation function (used when training model through backprop)
#calculation of backprop and also bashing out partial derivatives shows that dL/dw1 (the change in loss over the change in the first weight) changes by a very small amount
def derivSig(x):
    f = sigFunc(x)
    return f * (1-f)
    
#calculate potential loss
def loss(realY, predY):
    return ((realY - predY) ** 2).mean()
#initialize neuron
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def feed(self, inputs):
        #total is the dot product of weights and inputs (w1*i1) + (w2*i2) + ... + (wn*in) plus the bias b
        total = np.dot(self.weights, inputs) + self.bias
        #run through sigmoid to transform total to be passable into the next layer (range from 0 to 1, uninclusive)
        return sigFunc(total)
class NN:
    def __init__(self):
        #random weights from Normal distribution
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        #random biases from Normal distribution
        self.b1 = np.random.normal()  
        self.b2 = np.random.normal()  
        self.b3 = np.random.normal()  
    def feed(self, x):
        #x is an array with 2 elements (hidden nodes)
        h1 = sigFunc(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigFunc(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigFunc(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    def train(self, data, all_realY):
        #using a learning rate constant
        l_rate = 0.1
        #number of times we want to loop through a data set (better accuracy)
        epoch = 1000
        for i in range(epoch):
            for x, y in zip(data, all_realY):
                #manually feed forward since we will need h1 and h2 later
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigFunc(sum_h1)
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigFunc(sum_h2)
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigFunc(sum_o1)
                predY = o1
                
                #partial derivs
                #o1 neuron
                dL_dyPred  = -2 * (y - predY)
                dyPred_dw5 = h1 * derivSig(sum_o1)
                dyPred_dw6 = h2 * derivSig(sum_o1)
                dyPred_db3 = derivSig(sum_o1)
                dyPred_dh1 = self.w5 * derivSig(sum_o1)
                dyPred_dh2 = self.w6 * derivSig(sum_o1)
                #h1 neuron
                dh1_dw1 = x[0] * derivSig(sum_h1)
                dh1_dw2 = x[1] * derivSig(sum_h1)
                dh1_db1 = derivSig(sum_h1)
                #h2 neuron
                dh2_dw3 = x[0] * derivSig(sum_h2)
                dh2_dw4 = x[1] * derivSig(sum_h2)
                dh2_db2 = derivSig(sum_h2)

                #UPD weights and biases
                #neuron h1
                self.w1 -= l_rate * dL_dyPred * dyPred_dh1 * dh1_dw1
                self.w2 -= l_rate * dL_dyPred * dyPred_dh1 * dh1_dw2
                self.b1 -= l_rate * dL_dyPred * dyPred_dh1 * dh1_db1
                #neuron h2
                self.w3 -= l_rate * dL_dyPred * dyPred_dh2 * dh2_dw3
                self.w4 -= l_rate * dL_dyPred * dyPred_dh2 * dh2_dw4
                self.b2 -= l_rate * dL_dyPred * dyPred_dh2 * dh2_db2
                #neuron o1
                self.w5 -= l_rate * dL_dyPred * dyPred_dw5
                self.w6 -= l_rate * dL_dyPred * dyPred_dw6
                self.b3 -= l_rate * dL_dyPred * dyPred_db3
            #calc total loss at end of each epoch
            if i % 10 == 0:
                predY1 = np.apply_along_axis(self.feed, 1, data)
                los = loss(all_realY, predY1)
                print("Epoch %d loss: %.3f" % (i, los))
    
#define example data set 
#predict whether or not daniel's girlfriend is cheating on him based on text frequency and other
data = np.array([[90, 6], [10, 2], [50, 10], [2, 0], [100, 24], [0, 0]])
all_realY = np.array([1, 0, 1, 0, 1, 0])

network = NN()
network.train(data, all_realY)

daniel = np.array([70, 15])
kyle = np.array([10, 1])
print("Daniel: %.3f" % network.feed(daniel))
print("Kyle: %.3f" % network.feed(kyle)) 