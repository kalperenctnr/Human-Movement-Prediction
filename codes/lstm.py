import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
import seaborn as sn




def sigmoid(x):
    np.clip(x, -300, 300, out=x)
    return 1/(1 + np.exp(-x))

        
        
class LSTM:
    def __init__(self, hidden_size, input_size, output_size):
        '''
        LSTM to perform many-to-one tasks
        '''
        self.hidden_size = hidden_size
        
        w0_w = np.sqrt(6/(hidden_size+hidden_size))
        w0_u = np.sqrt(6/(input_size+hidden_size))
        w0_y = np.sqrt(6/(hidden_size+output_size))
        # Forget gate
        self.w_f = np.random.uniform(-w0_w, w0_w, (hidden_size, hidden_size))
        self.u_f = np.random.uniform(-w0_u, w0_u, (input_size, hidden_size))
        self.b_f = np.ones((1, hidden_size))
        
        # Input gate
        self.w_i = np.random.uniform(-w0_w, w0_w, (hidden_size, hidden_size))
        self.u_i = np.random.uniform(-w0_u, w0_u, (input_size, hidden_size))
        self.b_i = np.ones((1, hidden_size))
        
        # Output gate
        self.w_o = np.random.uniform(-w0_w, w0_w, (hidden_size, hidden_size))
        self.u_o = np.random.uniform(-w0_u, w0_u, (input_size, hidden_size))
        self.b_o = np.ones((1, hidden_size))
        
        # Candidate state
        self.w_g = np.random.uniform(-w0_w, w0_w, (hidden_size, hidden_size))
        self.u_g = np.random.uniform(-w0_u, w0_u, (input_size, hidden_size))
        self.b_g = np.ones((1, hidden_size))
        
        # Final output
        self.w_y = np.random.uniform(-w0_y, w0_y, (hidden_size, output_size))
        self.b_y = np.random.uniform(-w0_y, w0_y, (1, output_size))
        
        self.wf_cont = []
        self.uf_cont = []
        self.bf_cont = []
        self.wi_cont = []
        self.ui_cont = []
        self.bi_cont = []
        self.wo_cont = []
        self.uo_cont = []
        self.bo_cont = []
        self.wg_cont = []
        self.ug_cont = []
        self.bg_cont = []
        self.wy_cont = []
        self.by_cont = []
        
        # Momentums
        self.v_wf = np.zeros((hidden_size, hidden_size))
        self.v_uf = np.zeros((input_size, hidden_size))
        self.v_bf = np.zeros((1, hidden_size))
        
        self.v_wi = np.zeros((hidden_size, hidden_size))
        self.v_ui = np.zeros((input_size, hidden_size))
        self.v_bi = np.zeros((1, hidden_size))
        
        self.v_wo = np.zeros((hidden_size, hidden_size))
        self.v_uo = np.zeros((input_size, hidden_size))
        self.v_bo = np.zeros((1, hidden_size))
        
        self.v_wg = np.zeros((hidden_size, hidden_size))
        self.v_ug = np.zeros((input_size, hidden_size))
        self.v_bg = np.zeros((1, hidden_size))
        
        self.v_wy = np.zeros((hidden_size, output_size))
        self.v_by = np.zeros((1, output_size))
        
    
    def forward(self, inputs):
        '''
        Perform a forward pass of the LSTM.
        - inputs is a list of time steps for one input(one period) 
        - each vector is of the shape (1, input_size)
        '''
        # h: out state c: internal state(cell state)
        h = np.zeros((1, self.hidden_size))
        c = np.zeros((1, self.hidden_size))
        
        self.last_inputs = inputs
        self.last_h = { 0: h }
        self.last_c = { 0: c }
        self.last_f = {}
        self.last_i = {}
        self.last_o = {}
        self.last_g = {}
        
        for j, x in enumerate(inputs):
            f = sigmoid(x @ self.u_f + self.last_h[j] @ self.w_f + self.b_f)
            i = sigmoid(x @ self.u_i + self.last_h[j] @ self.w_i + self.b_i)
            o = sigmoid(x @ self.u_o + self.last_h[j] @ self.w_o + self.b_o)
            g = np.tanh(x @ self.u_g + self.last_h[j] @ self.w_g + self.b_g)
            self.last_f[j] = f
            self.last_i[j] = i
            self.last_o[j] = o
            self.last_g[j] = g
            
            c = f * self.last_c[j] + i * g
            h = o * np.tanh(c)
            self.last_h[j+1] = h
            self.last_c[j+1] = c
        
        # Output of LSTM
        y = h @ self.w_y + self.b_y
        return y
        
   
    def backprop(self, dy):
        '''
        Perform a backward pass of the LSTM.
        - d_y (dL/dy) has shape (1, output_size).
        '''
        
        n = len(self.last_inputs)
        
        # Gradients of final output part of lstm
        dw_y = self.last_h[n].T @ dy
        db_y = dy
        
        dw_f = np.zeros(self.w_f.shape)
        du_f = np.zeros(self.u_f.shape)
        db_f = np.zeros(self.b_f.shape)
        
        dw_i = np.zeros(self.w_i.shape)
        du_i = np.zeros(self.u_i.shape)
        db_i = np.zeros(self.b_i.shape)
        
        dw_o = np.zeros(self.w_o.shape)
        du_o = np.zeros(self.u_o.shape)
        db_o = np.zeros(self.b_o.shape)
        
        dw_g = np.zeros(self.w_g.shape)
        du_g = np.zeros(self.u_g.shape)
        db_g = np.zeros(self.b_g.shape)
        
        # Calculate dL/dh for the last h and c.
        # dL/dh = dL/dy * dy/dh
        d_h = dy @ self.w_y.T

        for t in reversed(range(n)):
            d_c = d_h * (self.last_o[t] * (1 - np.tanh(self.last_c[t+1]))) 
            d_g = d_h * self.last_i[t]
            d_o = d_h * np.tanh(self.last_c[t+1])
            d_i = d_c * self.last_g[t]
            d_f = d_c * self.last_c[t]
            
            dw_i += self.last_h[t].T @ (d_i * (1 - self.last_i[t]) * self.last_i[t]) 
            db_i += d_i * (1 - self.last_i[t]) * self.last_i[t]
            du_i += self.last_inputs[t].T @ (d_i * (1 - self.last_i[t]) * self.last_i[t]) 
            
            dw_f += self.last_h[t].T @ (d_f * (1 - self.last_f[t]) * self.last_f[t]) 
            db_f += d_f * (1 - self.last_f[t]) * self.last_f[t]
            du_f += self.last_inputs[t].T @ (d_f * (1 - self.last_f[t]) * self.last_f[t])
            
            dw_o += self.last_h[t].T @ (d_o * (1 - self.last_o[t]) * self.last_o[t])
            db_o += d_o * (1 - self.last_o[t]) * self.last_o[t]
            du_o += self.last_inputs[t].T @ (d_o * (1 - self.last_o[t]) * self.last_o[t])
            
            dw_g += self.last_h[t].T @ (d_g * (1 - self.last_g[t]**2))
            db_g += (d_g * (1 - self.last_g[t]**2))
            du_g += self.last_inputs[t].T @ (d_g * (1 - self.last_g[t]**2))
            

            d_h =  d_f * (self.last_f[t]) * (1 - self.last_f[t]) @ self.w_f  + d_o * (self.last_o[t]) * (1 - self.last_o[t]) @ self.w_o + d_i * (self.last_i[t]) * (1 - self.last_i[t]) @ self.w_i + d_g * (1 - self.last_g[t]**2) @ self.w_g 
        

        self.wf_cont.append(dw_f)
        self.uf_cont.append(du_f)
        self.bf_cont.append(db_f)
        self.wi_cont.append(dw_i)
        self.ui_cont.append(du_i)
        self.bi_cont.append(db_i)
        self.wo_cont.append(dw_o)
        self.uo_cont.append(du_o)
        self.bo_cont.append(db_o)
        self.wg_cont.append(dw_g)
        self.ug_cont.append(du_g)
        self.bg_cont.append(db_g)
        self.wy_cont.append(dw_y)
        self.by_cont.append(db_y)

        
    def update_weights(self, learn_rate=2e-2, batch_size=1, u=0.9):  
        '''
        Update weights after completing the mini-batch cycle
        '''
        self.v_wf = self.v_wf*u + learn_rate * (1/batch_size) * sum(self.wf_cont)
        self.w_f -= self.v_wf
        self.v_uf = self.v_uf*u + learn_rate * (1/batch_size) * sum(self.uf_cont)
        self.u_f -= self.v_uf
        self.v_bf = self.v_bf*u + learn_rate * (1/batch_size) * sum(self.bf_cont)
        self.b_f -= self.v_bf
        
        self.v_wi = self.v_wi*u + learn_rate * (1/batch_size) * sum(self.wi_cont)
        self.w_i -= self.v_wi
        self.v_ui = self.v_ui*u + learn_rate * (1/batch_size) * sum(self.ui_cont)
        self.u_i -= self.v_ui
        self.v_bi = self.v_bi*u + learn_rate * (1/batch_size) * sum(self.bi_cont)
        self.b_i -= self.v_bi
        
        self.v_wo = self.v_wo*u + learn_rate * (1/batch_size) * sum(self.wo_cont)
        self.w_o -= self.v_wo
        self.v_uo = self.v_uo*u + learn_rate * (1/batch_size) * sum(self.uo_cont)
        self.u_o -= self.v_uo
        self.v_bo = self.v_bo*u + learn_rate * (1/batch_size) * sum(self.bo_cont)
        self.b_o -= self.v_bo
        
        self.v_wg = self.v_wg*u + learn_rate * (1/batch_size) * sum(self.wg_cont)
        self.w_g -= self.v_wg
        self.v_ug = self.v_ug*u + learn_rate * (1/batch_size) * sum(self.ug_cont)
        self.u_g -= self.v_ug
        self.v_bg = self.v_bg*u + learn_rate * (1/batch_size) * sum(self.bg_cont)
        self.b_g -= self.v_bg
        
        self.v_wy = self.v_wy*u + learn_rate * (1/batch_size) * sum(self.wy_cont)
        self.w_y -= self.v_wy
        self.v_by = self.v_by*u + learn_rate * (1/batch_size) * sum(self.by_cont)
        self.b_y -= self.v_by
        
        # Containers re-initialization for next mini-batch updates
        self.wf_cont = []
        self.uf_cont = []
        self.bf_cont = []
        self.wi_cont = []
        self.ui_cont = []
        self.bi_cont = []
        self.wo_cont = []
        self.uo_cont = []
        self.bo_cont = []
        self.wg_cont = []
        self.ug_cont = []
        self.bg_cont = []
        self.wy_cont = []
        self.by_cont = []
        return 





file = h5.File('C:/Users/Sanyu Buke Badrac/Desktop/EEE 443/assignment3/codes/assign3_data3.h5', "r")
tr_x = np.array(file["trX"])
tr_y = np.array(file["trY"])
test_x = np.array(file["tstX"])
test_y = np.array(file["tstY"])



def createInputs(data):
  '''
  - Craetes inout data that are a list of vectors of the shape [input size, 1]
  '''
  inputs = []
  for row in data:
      a = row[:, None]
      inputs.append(a.T)
  return inputs



def softmax(xs):
  '''
  - Numerically robust softmax it uses basic exponnetial rule for robustness
  '''
  z = xs - np.max(xs)
  num = np.exp(z)
  den = np.sum(num)
  return num/den



def validation_loss(val_x, val_y):
    '''
    - Computes the mean of validation loss for the validation set
    '''
    val_loss = 0
    for i in range(300):
        val_inputs = createInputs(val_x[i])
        val_target = val_y[i][:, None]
        
        val_out = lstm.forward(val_inputs)
        # val_fc1_out = fc1.forward_propagation(out)
        # val_fc1_out_act = fc1_act.forward_propagation(val_fc1_out)
        # val_fc2_out = fc2.forward_propagation(val_fc1_out_act)
        val_probs = softmax(val_out)
        val_loss += (np.log(val_probs) @ -val_target)[0][0]
    return val_loss/300


def train_val_set(tr_x, tr_y):
    '''
    - Divides the training dataset into traning and validation sets, it randomly aranges them 
    '''
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


def predict(test_x):
    '''
    - Given a sample it predicts the class of it in a one-hot-coding form
    '''
    size = test_x.shape[0]
    predictions = np.zeros((6, size))
    for i in range(size):
        inputs = createInputs(test_x[i])
           
        out = lstm.forward(inputs)
        probs = softmax(out)
        index = np.argmax(probs)
        predictions[index, i] = 1
    return predictions



def confusion_matrix(y, predictions):
    '''
    - Given a predicted and actual label arrays, it construct a confusion matrix
    '''
    n = y.shape[0]
    c_m = np.zeros((6, 6))
    real = np.argmax(y, axis=1)
    predicted = np.argmax(predictions, axis=0)
    
    for i in range(n):
        c_m[predicted[i],real[i]] += 1
    return c_m



train_x, train_y, val_x, val_y = train_val_set(tr_x, tr_y)

lstm = LSTM(128, 3, 6)


acc_test = []
train_loss = []
val_loss = []
stop = False
for e in range(50):
    if not stop:
        shuffler = np.arange(2700)
        np.random.shuffle(shuffler)
        trn_x = np.array(train_x[shuffler])
        trn_y = np.array(train_y[shuffler])
        
        loss = 0
        for i in range(2700):
    
            inputs = createInputs(trn_x[i])
            target = trn_y[i][:, None]
           
            out = lstm.forward(inputs)

            probs = softmax(out)

            
            loss += (np.log(probs) @ -target)[0][0]
            d_L_d_y = -(target.T - probs)
            lstm.backprop(d_L_d_y)
            
            # Updating for the batch
            if i%32==31:
                lstm.update_weights(learn_rate=0.1, batch_size=32, u=0.85)
            if i==2699:
                lstm.update_weights(learn_rate=0.1, batch_size=12, u=0.85)
        epoch_loss = validation_loss(val_x, val_y)

        if epoch_loss < 0.8:
            stop = True

            
        train_loss.append(loss/2700)
        val_loss.append(epoch_loss)   
        
        predictions = predict(test_x)
    
        c_m = confusion_matrix(test_y, predictions)
        print(e)
        acc = 0
        for k in range(6):
            acc += c_m[k,k]
            
        acc_test.append(acc/600)

   
plt.plot(val_loss, "r", label="Val. Loss")
plt.plot(train_loss, "b", label="Train Loss")
plt.title("Cross-Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

predictions = predict(test_x)

c_m = confusion_matrix(test_y, predictions)
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

plt.plot(acc_test)
plt.title("Test Accuracy")
plt.show()