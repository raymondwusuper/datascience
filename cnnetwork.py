#This convolutional neural network is built from scratch, and does not use any specific ML libraries like tensorflow. 
#Therefore, albeit the network is relatively inefficient, it performs rather well, yielding around an 80-90% accuracy when tested with mnist data.
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

        #cache the input
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))
        for image_region, i, j in self.regions(input):
            output[i, j] = np.sum(image_region * self.filters, axis = (1, 2))
            #axis = (1, 2) produces a 1d array of length num_filters with each element containing the convolution result for the corresponding filter
        #this returns a 3d np array with dimensions num_filters x h x w
        return output
    
    def backprop(self, dL_dout, learn_rate):
        #backwards pass through the CNN layers
        #dL_dout is the gradient for this output
        #the dl_dfilters is just the sum of all corresponding image pixel values
        dl_dfilters = np.zeros(self.filters.shape)

        for image_region, i, j in self.regions(self.last_input):
            for f in range(self.num_filters):
                dl_dfilters[f] += dL_dout[i, j, f] * image_region

        #update filters
        self.filters -= learn_rate * dl_dfilters
        #no return because this is our first layer in the CNN, otherwise we would just return the loss gradient
        return None

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

        #cache our input
        self.last_input = input

        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters))

        #puts the input into regions and adds it to output np array
        for image_region, i, j in self.regions(input):
            output[i, j] = np.amax(image_region, axis = (0, 1))
        
        return output
    
    def backprop(self, dL_dout):
        #backwards pass through the maxpool layer
        #dL_dout is the loss gradient for this layer
        dl_dinput = np.zeros(self.last_input.shape)
        for image_region, i, j in self.regions(self.last_input):
            h, w, f = image_region.shape
            arrMax = np.amax(image_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        #if this pixel in the input is the max, copy the gradient to it
                        if image_region[i2, j2, f2] == arrMax[f2]:
                            #recall that we divided by 2 to read the input shape
                            #we will therefore remultiply by 2 to map the gradient correctly to our dl_dinput array
                            dl_dinput[i*2 + i2, j*2 + j2, f2] = dL_dout[i, j, f2]
        
        return dl_dinput

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
        self.last_input_shape = input.shape
        #cache the values inputted
        input = input.flatten()
        self.last_input = input

        inputLength, nodes = self.weights.shape
        total = np.dot(input, self.weights) + self.bias
        self.last_total = total
        exp = np.exp(total)
        return exp/np.sum(exp, axis = 0)
    
    def backprop(self, dL_dout, learn_rate):
        #backwards pass through the softmax layer
        #we need the gradients to minimize loss per output
        for i, grad in enumerate(dL_dout):
            if grad == 0:
                continue
            total_exp = np.exp(self.last_total)
            #sum of all e^totals
            skibidi = np.sum(total_exp)

            #calculate gradient of out[i] against totals
            dout_dT = -total_exp[i] * total_exp / (skibidi ** 2)
            dout_dT[i] = total_exp[i] * (skibidi - total_exp) / (skibidi **2)

            #gradients of totals vs weights dot input + biases
            dt_dw = self.last_input
            dt_db = 1
            dt_dInputs = self.weights
            #loss v totals
            #precalculating dL_dt because we use it to calculate gradients
            dL_dt = grad * dout_dT
            #loss v weights/biases/inputs
            #@ is matrix multiplication
            #np.newaxis creates new axes of length 1 to multiply our matrices
            #our final result will have dimensions inputLength * nodes
            dl_dw = dt_dw[np.newaxis].T @ dL_dt[np.newaxis]
            dl_db = dL_dt * dt_db
            #multiply matrices with inputLength * nodes by nodes * 1 to get an array with just length inputLength
            dl_dinputs = dt_dInputs @ dL_dt

            #update weights to train
            self.weights -= learn_rate * dl_dw
            self.bias -= learn_rate * dl_db
            #we must reshape this output to the same as the cached input because we flattened it earlier in the feed() function
            return dl_dinputs.reshape(self.last_input_shape)
        


from mnist import MNIST

#for some reason, after running the command sequence cd python-mnist and ./bin/mnist_get_data.sh* the data still doesn't appear like it should
#this is a problem with my current computer and im pretty sure it will work on another device.
mndata = MNIST('data/')
mndata.gz = False
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing() 

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

def train(image, label, learning_rate=.005):
    #completes a full train
    #image is a 2d np array and label is a digit
    out, loss, acc = feed(image, label)
    #initial gradient
    grad = np.zeros(10)
    grad[label] = -1 / out[label]
    #backward feed the gradient to minimize
    grad = softmax.backprop(grad, learning_rate)
    grad = pool.backprop(grad)
    grad = network.backprop(grad, learning_rate)
    #note that the previous line sets grad to None. Once we run it through the network to train it, we dont need grad anymore
    return loss, acc

for epoch in range(3):
    print('Epoch %d:' % (epoch + 1))
    #randomize training data
    permute = np.random.permutation(len(train_images))
    train_images = train_images[permute]
    train_labels = train_labels[permute]
    #train the network
    loss = 0
    correct = 0
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i & 100 == 99:
            #view how loss diminishes as we perform a forward pass through the network
            print("[Step %d] past 100 steps: Average loss %.3f | Accuracy %d%%" % (i+1, loss/100, correct))
            loss = 0
            correct = 0
        #adds values to the loss and accuracies accordingly and calls train again
        l, acc = train(image, label)
        loss += l
        correct += acc

#testing
loss = 0
correct = 0
for image, label in zip(test_images, test_labels):
    _, l, acc = feed(image, label)  
    loss += l
    correct += acc
num_tests = len(test_images)
print("Test Loss:", loss / num_tests)
print("Test Accuracy:", correct / num_tests)  
            
