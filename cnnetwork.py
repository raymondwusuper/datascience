import numpy as np
#class using 3x3 filters
class CNN:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        #filters is a 3d array with dimensions numfiltersx3x3
        #divide by 9 to reduce variance
        self.filters = np.random.randn(num_filters, 3, 3) / 9
    #create a helper method to generate all 3x3 image regions
    def regions(self, image):
        #image is a 2d numpy array
        h, w = image.shape
        #takes regions of the input image
        for i in range(h-2):
            for j in range(w-2):
                image_region = image[i:i+3, j:j+3]
                yield image_region, i, j
    def feed(self, input):
        #feeds input through convolutional layer
        #input is 2d np array
        h, w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))
        for image_region, i, j in self.regions(input):
            output[i, j] = np.sum(image_region * self.filters, axis = (1, 2))
            #axis = (1, 2) produces a 1d array of length num_filters with each element containing the convolution result for the corresponding filter
        #this returns a 3d np array with dimensions num_filters x h x w
        return output

#pool the previous 3d array into a 2d array using pooling
class maxPool:
    def regions(self, image):
        #generate non-overlapping regions to pool
        h, w, _ = image.shape #2d np array
        new_h = h//2
        new_w = w//2

        #generate 2x2 regions to scan over
        for i in range(new_h):
            for j in range(new_w):
                image_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield image_region, i, j
        
    def feed(self, input):
        #shape is a 3d np array
        #feeds the pooled layer using input
        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters))

        #puts the input into regions and adds it to output np array
        for image_region, i, j in self.regions(input):
            output[i, j] = np.amax(image_region, axis = (0, 1))
        
        return output

#condenses data into 10 nodes which will help find the digit with the highest probability
#this will be the output of CNN
#uses cross entropy loss which holds previous predictions as a standard for loss calculation
class softMax:
    def __init__(self, inputLength, nodes):
        #im just going to copy this stuff from my first neural network
        #just weights and biases, standard neural network stuff
        self.weights = np.random.randn(inputLength, nodes) / inputLength
        self.bias = np.zeros(nodes)

    def feed(self, input):
        #feeds the softmax layer using input, returns a 1d np array with respective probability values
        input = input.flatten()

        inputLength, nodes = self.weights.shape
        total = np.dot(input, self.weights) + self.bias
        exp = np.exp(total)
        return exp/np.sum(exp, axis = 0)

from mnist import MNIST

mndata = MNIST('data/')
mndata.gz = False
images, labels = mndata.load_testing() #for some reason, after running the command sequence cd python-mnist and ./bin/mnist_get_data.sh* the data still doesn't appear like it should
                                       #this is a problem with my current computer and im pretty sure it will work on another device.

network = CNN(8)
pool = maxPool()
softmax = softMax(13*13*8, 10)

def feed(image, label):
    #passes the cnn and calculates loss
    #image is a 2d np array, label is a digit

    out = network.feed((image / 255) - 0.5)
    out = pool.feed(out)
    out = softmax.feed(out)

    #calculate loss and accuracy
    loss = -np.log(out[label]) #natural log because of how cross entropy loss is calculated (L = -ln(p_c))
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

loss = 0
correct = 0
for i, (image, label) in enumerate(zip(images, labels)):
    _, l, acc = feed(image, label)
    loss += l
    correct += acc
    #adds values to the loss and accuracies accordingly

    if i % 100 == 99:
        print("[Step %d] past 100 steps: Average loss %.3f | Accuracy %d%%" % (i+1, loss/100, correct))
        #view how loss diminishes as we perform a forward pass through the network
        loss = 0
        correct = 0
