import h5py as h5
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn

def sigmoid(x):
    return 1/(1 + np.exp(-x))
 
        
class GRU:
    def __init__(self, hidden_size, input_size, output_size):
        self.hidden_size = hidden_size
        
        w0_w = np.sqrt(6/(hidden_size+hidden_size))
        w0_u = np.sqrt(6/(input_size+hidden_size))
        w0_y = np.sqrt(6/(hidden_size+output_size))
        # Forget gate
        self.w_f = np.random.uniform(-w0_w, w0_w, (hidden_size, hidden_size))
        self.u_f = np.random.uniform(-w0_u, w0_u, (input_size, hidden_size))
        self.b_f = np.ones((1, hidden_size))
        
        # Reset Gate
        self.w_r = np.random.uniform(-w0_w, w0_w, (hidden_size, hidden_size))
        self.u_r = np.random.uniform(-w0_u, w0_u, (input_size, hidden_size))
        self.b_r = np.ones((1, hidden_size))
        
        # Update Gate
        self.w_z = np.random.uniform(-w0_w, w0_w, (hidden_size, hidden_size))
        self.u_z = np.random.uniform(-w0_u, w0_u, (input_size, hidden_size))
        self.b_z = np.ones((1, hidden_size))
        
        # Current state
        self.w_c = np.random.uniform(-w0_w, w0_w, (hidden_size, hidden_size))
        self.u_c = np.random.uniform(-w0_u, w0_u, (input_size, hidden_size))
        self.b_c = np.ones((1, hidden_size))
        
        # Final output
        self.w_y = np.random.uniform(-w0_y, w0_y, (hidden_size, output_size))
        self.b_y = np.ones((1, output_size))
        
        self.wr_cont = []
        self.ur_cont = []
        self.br_cont = []
        self.wz_cont = []
        self.uz_cont = []
        self.bz_cont = []
        self.wc_cont = []
        self.uc_cont = []
        self.bc_cont = []
        self.wy_cont = []
        self.by_cont = []
        
        # Momentums
        self.v_wr = np.zeros((hidden_size, hidden_size))
        self.v_ur = np.zeros((input_size, hidden_size))
        self.v_br = np.zeros((1, hidden_size))
        
        self.v_wz = np.zeros((hidden_size, hidden_size))
        self.v_uz = np.zeros((input_size, hidden_size))
        self.v_bz = np.zeros((1, hidden_size))
    
        self.v_wc = np.zeros((hidden_size, hidden_size))
        self.v_uc = np.zeros((input_size, hidden_size))
        self.v_bc = np.zeros((1, hidden_size))
                
        self.v_wy = np.zeros((hidden_size, output_size))
        self.v_by = np.zeros((1, output_size))
        
    # computes the output Y of a layer for a given input X
    def forward(self, inputs):
        # h: out state c: current state
        h = np.zeros((1, self.hidden_size))

        
        self.last_inputs = inputs
        self.last_h = { 0: h }
        self.last_r = {}
        self.last_z = {}
        self.last_c = {}
        
        for j, x in enumerate(inputs):
            r = sigmoid(x @ self.u_r + self.last_h[j] @ self.w_r + self.b_r)
            z = sigmoid(x @ self.u_z + self.last_h[j] @ self.w_z + self.b_z)
            c = np.tanh(x @ self.u_c + (r *self.last_h[j]) @ self.w_c + self.b_c)

            self.last_r[j] = r
            self.last_z[j] = z
            self.last_c[j] = c
            
            h = (1 - z) * c + z * (self.last_h[j])
            self.last_h[j+1] = h
        
        # Output of GRU
        y = h @ self.w_y + self.b_y
        return y
        
    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backprop(self, dy):
        '''
        Perform a backward pass of the LSTM.
        - d_y (dL/dy) has shape (1, output_size).
        '''
        
        n = len(self.last_inputs)
        
        # Gradients of final output part of lstm
        dw_y = self.last_h[n].T @ dy
        db_y = dy
        
        dw_r = np.zeros(self.w_r.shape)
        du_r = np.zeros(self.u_r.shape)
        db_r = np.zeros(self.b_r.shape)
        
        dw_z = np.zeros(self.w_z.shape)
        du_z = np.zeros(self.u_z.shape)
        db_z = np.zeros(self.b_z.shape)
        
        dw_c = np.zeros(self.w_c.shape)
        du_c = np.zeros(self.u_c.shape)
        db_c = np.zeros(self.b_c.shape)
        

        
        # Calculate dL/dh for the last h and c.
        # dL/dh = dL/dy * dy/dh
        d_h = dy @ self.w_y.T

        for t in reversed(range(n)):
            # Computing gate gradients 
            d_c = d_h * (1 - self.last_z[t]) 
            d_r = d_c * (1 - self.last_c[t]**2) * self.last_r[t] @ self.w_c
            d_z = d_h * (-self.last_c[t] + self.last_h[t])

            # Calculating weight gradients
            dw_r += self.last_h[t].T @ (d_r * (1 - self.last_r[t]) * self.last_r[t]) 
            db_r += d_r * (1 - self.last_r[t]) * self.last_r[t]
            du_r += self.last_inputs[t].T @ (d_r * (1 - self.last_r[t]) * self.last_r[t]) 
            
            dw_z += self.last_h[t].T @ (d_z * (1 - self.last_z[t]) * self.last_z[t]) 
            db_z += d_z * (1 - self.last_z[t]) * self.last_z[t]
            du_z += self.last_inputs[t].T @ (d_z * (1 - self.last_z[t]) * self.last_z[t])
            
            dw_c += (self.last_r[t] * self.last_h[t]).T @ (d_c * (1 - self.last_c[t]**2))
            db_c += (d_c * (1 - self.last_c[t]**2))
            du_c += self.last_inputs[t].T @ (d_c * (1 - self.last_c[t]**2))
            
            # Passing the state recursively
            d_h =  d_r * (self.last_r[t]) * (1 - self.last_r[t]) @ self.w_r  + d_z * (self.last_z[t]) * (1 - self.last_z[t]) @ self.w_z + d_h * self.last_z[t] + d_c * (1 - self.last_c[t]**2) * self.last_r[t] @ self.w_c 
        

        self.wr_cont.append(dw_r)
        self.ur_cont.append(du_r)
        self.br_cont.append(db_r)
        self.wz_cont.append(dw_z)
        self.uz_cont.append(du_z)
        self.bz_cont.append(db_z)
        self.wc_cont.append(dw_c)
        self.uc_cont.append(du_c)
        self.bc_cont.append(db_c)
        self.wy_cont.append(dw_y)
        self.by_cont.append(db_y)

        
    def update_weights(self, learn_rate=2e-2, batch_size=1, u=0.9):  
        
        self.v_wr = self.v_wr*u + learn_rate * (1/batch_size) * sum(self.wr_cont)
        self.w_r -= self.v_wr
        self.v_ur = self.v_ur*u + learn_rate * (1/batch_size) * sum(self.ur_cont)
        self.u_r -= self.v_ur
        self.v_br = self.v_br*u + learn_rate * (1/batch_size) * sum(self.br_cont)
        self.b_r -= self.v_br
        
        self.v_wz = self.v_wz*u + learn_rate * (1/batch_size) * sum(self.wz_cont)
        self.w_z -= self.v_wz
        self.v_ui = self.v_uz*u + learn_rate * (1/batch_size) * sum(self.uz_cont)
        self.u_z -= self.v_uz
        self.v_bz = self.v_bz*u + learn_rate * (1/batch_size) * sum(self.bz_cont)
        self.b_z -= self.v_bz
        
        self.v_wc = self.v_wc*u + learn_rate * (1/batch_size) * sum(self.wc_cont)
        self.w_c -= self.v_wc
        self.v_uc = self.v_uc*u + learn_rate * (1/batch_size) * sum(self.uc_cont)
        self.u_c -= self.v_uc
        self.v_bc = self.v_bc*u + learn_rate * (1/batch_size) * sum(self.bc_cont)
        self.b_c -= self.v_bc
        
        
        self.v_wy = self.v_wy*u + learn_rate * (1/batch_size) * sum(self.wy_cont)
        self.w_y -= self.v_wy
        self.v_by = self.v_by*u + learn_rate * (1/batch_size) * sum(self.by_cont)
        self.b_y -= self.v_by
        
        # Containers re-initialization for next mini-batch updates
        self.wr_cont = []
        self.ur_cont = []
        self.br_cont = []
        self.wz_cont = []
        self.uz_cont = []
        self.bz_cont = []
        self.wc_cont = []
        self.uc_cont = []
        self.bc_cont = []
        self.wy_cont = []
        self.by_cont = []
        return 
    
    


file = h5.File('assign3_data3.h5', "r")
tr_x = np.array(file["trX"])
tr_y = np.array(file["trY"])
test_x = np.array(file["tstX"])
test_y = np.array(file["tstY"])


def createInputs(data):
  inputs = []
  for row in data:
      a = row[:, None]
      inputs.append(a.T)
  return inputs


def softmax(xs):
  z = xs - np.max(xs)
  num = np.exp(z)
  den = np.sum(num)
  return num/den


def validation_loss(val_x, val_y):
    val_loss = 0
    for i in range(300):
        val_inputs = createInputs(val_x[i])
        val_target = val_y[i][:, None]
        
        val_out = gru.forward(val_inputs)
        val_probs = softmax(val_out)
        val_loss += (np.log(val_probs) @ -val_target)[0][0]
    return val_loss/300


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


def predict(test_x):
    size = test_x.shape[0]
    predictions = np.zeros((6, size))
    for i in range(size):
        inputs = createInputs(test_x[i])
           
        out = gru.forward(inputs)
        probs = softmax(out)
        index = np.argmax(probs)
        predictions[index, i] = 1
    return predictions



def confusion_matrix(y, predictions):
    n = y.shape[0]
    c_m = np.zeros((6, 6))
    real = np.argmax(y, axis=1)
    predicted = np.argmax(predictions, axis=0)
    
    for i in range(n):
        c_m[predicted[i],real[i]] += 1
    return c_m



train_x, train_y, val_x, val_y = train_val_set(tr_x, tr_y)

gru = GRU(128, 3, 6)

inputs = createInputs(tr_x[0])
target = tr_y[0][:, None]

out = gru.forward(inputs)

gru.backprop(-(target.T - out))

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
           
            out = gru.forward(inputs)


            probs = softmax(out)

            
            loss += (np.log(probs) @ -target)[0][0]
            d_L_d_y = -(target.T - probs)
            gru.backprop(d_L_d_y)
            
            if i%32==31:
                gru.update_weights(learn_rate=0.1, batch_size=32, u=0.85)

            if i==2699:
                gru.update_weights(learn_rate=0.1, batch_size=12, u=0.85)
        epoch_loss = validation_loss(val_x, val_y)


        if epoch_loss < 0.5:
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