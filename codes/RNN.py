import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
import seaborn as sn

class RNN:

  def __init__(self, input_size, output_size, hidden_size=64):
    w0_hy = np.sqrt(6/(hidden_size+output_size))
    w0_xh = np.sqrt(6/(3+hidden_size))
    w0_hh = np.sqrt(6/(2*hidden_size))

    # Weights    
    self.Whh = np.random.uniform(-w0_hh, w0_hh, (hidden_size, hidden_size))
    self.Wxh = np.random.uniform(-w0_xh, w0_xh, (hidden_size, input_size))
    self.Why = np.random.uniform(-w0_hy, w0_hy, (output_size, hidden_size))

    
    self.bh = np.random.uniform(-w0_hh, w0_hh, (hidden_size, 1))
    self.by = np.random.uniform(-w0_hy, w0_hy, (output_size, 1))
    # Containers for mini-batch updates
    self.hh_cont = []
    self.xh_cont = []
    self.hy_cont = []
    self.bh_cont = []
    self.by_cont = []
    
    # Momentums
    self.v_hh = np.zeros((hidden_size, hidden_size))
    self.v_xh = np.zeros( (hidden_size, input_size))
    self.v_hy = np.zeros((output_size, hidden_size))
    self.v_bh = np.zeros((hidden_size, 1))
    self.v_by = np.zeros((output_size, 1))
    
  def forward(self, inputs):
    '''
    Perform a forward pass of the RNN using the given inputs.
    Returns the final output and hidden state.
    - inputs is an array of one hot vectors with shape (input_size, 1).
    '''
    h = np.zeros((self.Whh.shape[0], 1))

    self.last_inputs = inputs
    self.last_hs = { 0: h }

    # Perform each step of the RNN
    for i, x in enumerate(inputs):
      h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
      self.last_hs[i + 1] = h
    
    # Compute the output

    y = self.Why @ h + self.by

    return y, h

  def backprop(self, d_y):
    '''
    Perform a backward pass of the RNN.
    - d_y (dL/dy) has shape (output_size, 1).
    '''
    n = len(self.last_inputs)

    # Calculate dL/dWhy and dL/dby.
    d_Why = d_y @ self.last_hs[n].T
    d_by = d_y

    # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
    d_Whh = np.zeros(self.Whh.shape)
    d_Wxh = np.zeros(self.Wxh.shape)
    d_bh = np.zeros(self.bh.shape)

    # Calculate dL/dh for the last h.
    # dL/dh = dL/dy * dy/dh
    d_h = self.Why.T @ d_y

    # Backpropagate through time.
    # BPTT is truncated by 30 steps in order to prevent exploding/vanishing gradients
    for t in reversed(range(n)[120:]):
      # An intermediate value: dL/dh * (1 - h^2)
      temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

      # dL/db = dL/dh * (1 - h^2)
      d_bh += temp

      # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
      d_Whh += temp @ self.last_hs[t].T

      # dL/dWxh = dL/dh * (1 - h^2) * x
      d_Wxh += temp @ self.last_inputs[t].T

      # Next dL/dh = dL/dh * (1 - h^2) * Whh
      d_h = self.Whh @ temp

    # Clip to prevent exploding gradients.
    for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
      np.clip(d, -1, 1, out=d)
    
    self.hh_cont.append(d_Whh)
    self.xh_cont.append(d_Wxh)
    self.hy_cont.append(d_Why)
    self.bh_cont.append(d_bh)
    self.by_cont.append(d_by)
    

    
  def update_weights(self, learn_rate=2e-2, batch_size=1, u=0.9):  
    # Update weights and biases using mini-batch gd with momentum.
    self.v_hh = self.v_hh*u + learn_rate * (1/batch_size) * sum(self.hh_cont)
    self.Whh -= self.v_hh
    
    self.v_xh = self.v_xh*u + learn_rate * (1/batch_size) * sum(self.xh_cont)
    self.Wxh -= self.v_xh
    
    self.v_hy = self.v_hy*u + learn_rate * (1/batch_size) * sum(self.hy_cont)
    self.Why -= self.v_hy
    
    self.v_bh = self.v_bh*u + learn_rate * (1/batch_size) * sum(self.bh_cont)
    self.bh -= self.v_bh
    
    self.v_by = self.v_by*u + learn_rate * (1/batch_size) * sum(self.by_cont)
    self.by -= self.v_by
      
    # Containers re-initialization for next mini-batch updates
    self.hh_cont = []
    self.xh_cont = []
    self.hy_cont = []
    self.bh_cont = []
    self.by_cont = []
    return


# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
        

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        w0 = np.sqrt(6/(input_size+output_size))
        self.weights = np.random.uniform(-w0, w0, (output_size, input_size))
        self.bias = np.random.uniform(-w0, w0, (output_size, 1))
        # self.bias = np.zeros((output_size, 1))
        
        self.w_cont = []
        self.b_cont = []
        
        self.v_w = np.zeros(((output_size, input_size)))
        self.v_b = np.zeros((output_size, 1))
        
    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.weights @ input_data + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate=0.1):
        input_error = self.weights.T @ output_error
        dw =  output_error @ self.input.T
        db = output_error
        
        self.w_cont.append(dw)
        self.b_cont.append(db)
        return input_error
    
    def update_weights(self, batch_size=1, learn_rate=0.1, u=0.9):
        #self.v_w = self.v_w*u + learn_rate * (1/batch_size) * sum(self.w_cont)
        self.weights -= learn_rate * (1/batch_size) * sum(self.w_cont)
        
        #self.v_b = self.v_b*u + learn_rate * (1/batch_size) * sum(self.b_cont)
        self.bias -= learn_rate * (1/batch_size) * sum(self.b_cont)
        
        self.w_cont = []
        self.b_cont = []
        return

#from layer import Layer

# inherit from base class Layer
class ActivationLayer(Layer):
    
    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.input*(self.input>0)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate=0.1):
        return (1*(self.input>0)) * output_error



file = h5.File('C:/Users/Sanyu Buke Badrac/Desktop/EEE 443/assignment3/codes/assign3_data3.h5', "r")
tr_x = np.array(file["trX"])
tr_y = np.array(file["trY"])
test_x = np.array(file["tstX"])
test_y = np.array(file["tstY"])




def createInputs(data):

  inputs = []
  for row in data:
      a = row[:, None]
      inputs.append(a)
  return inputs

def softmax(xs):
  z = xs - max(xs)
  num = np.exp(z)
  den = np.sum(num)
  return num/den

def train_val_set(tr_x, tr_y):
    train_x = np.zeros((2700, 150, 3))
    train_y = np.zeros((2700, 6))
    val_x = np.zeros((300, 150, 3))
    val_y = np.zeros((300, 6))
    
    arr = np.arange(500)
    np.random.shuffle(arr)
    for i in range(6):
        train_x[450*i:450*(i+1)] = tr_x[i*500 + arr[0:450]]
        val_x[50*i:50*(i+1)] = tr_x[i*500 + arr[450:]]
        train_y[450*i:450*(i+1)] = tr_y[i*500 + arr[0:450]]
        val_y[50*i:50*(i+1)] = tr_y[i*500 + arr[450:]]
    return train_x,train_y,val_x,val_y


def validation_loss(val_x, val_y):
    val_loss = 0
    for i in range(300):
        val_inputs = createInputs(val_x[i])
        val_target = val_y[i][:, None]
        
        val_out,_ = rnn.forward(val_inputs)
        val_fc1_out = fc1.forward_propagation(out)
        val_fc1_out_act = fc1_act.forward_propagation(val_fc1_out)
        val_fc2_out = fc2.forward_propagation(val_fc1_out_act)
        val_probs = softmax(val_fc2_out)
        val_loss += (-val_target.T @ np.log(val_probs))[0][0]
    return val_loss/300



def predict(test_x):
    size = test_x.shape[0]
    predictions = np.zeros((size, 6))
    for i in range(size):
        inputs = createInputs(test_x[i])
        
        out,_ = rnn.forward(inputs)
        fc1_out = fc1.forward_propagation(out)
        fc1_out_act = fc1_act.forward_propagation(fc1_out)
        fc2_out = fc2.forward_propagation(fc1_out_act)
        probs = softmax(fc2_out)
        index = np.argmax(probs)
        predictions[i, index] = 1
    return predictions



def confusion_matrix(y, predictions):
    n = y.shape[0]
    c_m = np.zeros((6, 6))
    real = np.argmax(y, axis=1)
    predicted = np.argmax(predictions, axis=1)
    
    for i in range(n):
        c_m[predicted[i],real[i]] += 1
    return c_m



train_x, train_y, val_x, val_y = train_val_set(tr_x, tr_y)
rnn = RNN(3, 32, 128)
rrn1 = RNN(3, 32, 128)
fc1 = FCLayer(32, 64)
fc1_act = ActivationLayer()
fc2 = FCLayer(64, 6)

inputs = createInputs(tr_x[0])
target = tr_y[0][:, None]

out,_ = rnn.forward(inputs)
fc1_out = fc1.forward_propagation(out)
fc1_out_act = fc1_act.forward_propagation(fc1_out)
fc2_out = fc2.forward_propagation(fc1_out_act)
probs = softmax(fc2_out)

loss = (-target.T @ np.log(probs))[0][0]

err2 = fc2.backward_propagation(target - probs)
err1_act = fc1_act.backward_propagation(err2)
err1 = fc1.backward_propagation(err1_act)

d_L_d_y = err1
rnn.backprop(d_L_d_y)
rnn.update_weights()
fc1.update_weights()
fc2.update_weights()

# val_loss = validation_loss(val_x, val_y)

train_loss = []
val_loss = []
acc_test = []


stop = False
for e in range(50):
    if not stop:
        shuffler = np.arange(2700)
        np.random.shuffle(shuffler)
        trn_x = np.array(train_x[shuffler])
        trn_y = np.array(train_y[shuffler])
        
        loss = 0
        l_r = 0.1/(1 + e/5)
        for i in range(2700):
    
            inputs = createInputs(trn_x[i])
            target = trn_y[i][:, None]
           
            out,_ = rnn.forward(inputs)
            fc1_out = fc1.forward_propagation(out)
            fc1_out_act = fc1_act.forward_propagation(fc1_out)
            fc2_out = fc2.forward_propagation(fc1_out_act)
            probs = softmax(fc2_out)
            loss += (-target.T @ np.log(probs))[0][0]
            
            err2 = fc2.backward_propagation(-(target - probs))
            err1_act = fc1_act.backward_propagation(err2)
            err1 = fc1.backward_propagation(err1_act)
            
            d_L_d_y = err1
            rnn.backprop(d_L_d_y)
            
            
            
            if i%32==31:
                rnn.update_weights(learn_rate=l_r, batch_size=32, u=0.85)
                fc1.update_weights(32, learn_rate=2*l_r, u=0.85)
                fc2.update_weights(32, learn_rate=2*l_r, u=0.85)

            if i==2699:
                rnn.update_weights(learn_rate=l_r, batch_size=12, u=0.85)
                fc1.update_weights(32, learn_rate=2*l_r, u=0.85)
                fc2.update_weights(32, learn_rate=2*l_r, u=0.85)
        epoch_loss = validation_loss(val_x, val_y)
        print("Training Loss "+str(loss/2700))
        print("Validation Loss "+str(epoch_loss))

        if epoch_loss < 1.8:
            stop = True
            
                       
        predictions = predict(test_x)
    
        c_m = confusion_matrix(test_y, predictions)
        acc = 0
        for k in range(6):
            acc += c_m[k,k]
            
        acc_test.append(acc/600)
        print("-------------------------")
        print("Test Accuracy: "+str(acc/600))
        print("-------------------------")
        sn.heatmap(c_m,xticklabels=[1, 2, 3, 4, 5, 6], yticklabels=[1, 2, 3, 4, 5, 6], annot=True, fmt='g')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Confusion Matrix Test")
        plt.show()
        
        predictions = predict(train_x)
        c_m = confusion_matrix(train_y, predictions)
        
        sn.heatmap(c_m,xticklabels=[1, 2, 3, 4, 5, 6], yticklabels=[1, 2, 3, 4, 5, 6], annot=True, fmt='g')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Confusion Matrix Train")
        plt.show()
        
        
        predictions = predict(val_x)
        c_m = confusion_matrix(val_y, predictions)

        
        sn.heatmap(c_m,xticklabels=[1, 2, 3, 4, 5, 6], yticklabels=[1, 2, 3, 4, 5, 6], annot=True, fmt='g')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Confusion Matrix Validation")
        plt.show()
        train_loss.append(loss/2700)
        val_loss.append(epoch_loss)
    
       
plt.plot(val_loss, "r", label="Val. Loss")
plt.plot(train_loss, "b", label="Train Loss")
plt.title("Cross-Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


predictions = predict(test_x)

c_m = confusion_matrix(test_y, predictions)
print(c_m)

acc = 0
for k in range(6):
    acc += c_m[k,k]   
print("-------------------------")
print("Test Accuracy: "+str(acc/600))
print("-------------------------")


sn.heatmap(c_m,xticklabels=[1, 2, 3, 4, 5, 6], yticklabels=[1, 2, 3, 4, 5, 6], annot=True, fmt='g')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Confusion Matrix Test")
plt.show()

predictions = predict(train_x)
c_m = confusion_matrix(train_y, predictions)
print(c_m)

sn.heatmap(c_m,xticklabels=[1, 2, 3, 4, 5, 6], yticklabels=[1, 2, 3, 4, 5, 6], annot=True, fmt='g')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Confusion Matrix Train")
plt.show()







