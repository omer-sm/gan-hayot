import numpy as np
import matplotlib.pyplot as plt
import os 
import h5py
from sklearn.metrics import classification_report, confusion_matrix
import math
import matplotlib
from scipy import ndimage

IS_DEV = False
xp = np
cpx = None

class DLModel:

    def __init__(self, name="Model", inject_str_func=None, use_cuda=False):
        self.name = name
        self.layers = [None]
        self._is_compiled = False
        self.inject_str_func = inject_str_func
        self.use_cuda = use_cuda
        if use_cuda and IS_DEV:
            import cupy as cp
            import cupyx
            from cupyx.scipy import ndimage
            global xp
            xp = cp
            global cpx
            cpx = cupyx

    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers)-1) +"\n"
        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) +"\n"
            s += "\t\tloss function: " + self.loss + "\n\n"
        for i in range(1,len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"
        return s

    def add(self, layer):
        if type(layer) == DLLayer and layer._activation == "vae_bottleneck":
            self.bottleneck_layer = len(self.layers)
        self.layers.append(layer)

    def squared_means(self, AL, Y):
        return (Y-AL)**2

    def squared_means_backward(self, AL, Y):
        return -2*(Y-AL)

    def cross_entropy(self, AL, Y):
        AL = xp.where(AL == 0, AL+1e-10, AL )
        AL = xp.where(AL == 1, AL-1e-10, AL )
        return xp.where(Y == 0, -xp.log(1-AL), -xp.log(AL))

    def cross_entropy_backward(self, AL, Y):
        AL = xp.where(AL == 0, AL+1e-10, AL )
        AL = xp.where(AL == 1, AL-1e-10, AL )
        return xp.where(Y == 0, 1/(1-AL), -1/AL)

    def categorical_cross_entropy(self, AL, Y):
        AL = xp.where(AL <= 0, 1e-10, AL )
        return xp.where(Y == 1, -xp.log(AL), 0)

    def categorical_cross_entropy_backward(self, AL, Y):
        return AL - Y

    def cross_entropy_KLD(self, AL, Y):
        CE = self.cross_entropy(AL, Y) * self.recon_loss_weight
        logvar = self.layers[self.bottleneck_layer].logvar
        mu = self.layers[self.bottleneck_layer].mu
        KL = -xp.sum(1 + xp.log(logvar**2) - mu**2 - logvar**2)
        CE[0][0] += KL 
        return CE

    def cross_entropy_KLD_backward(self, AL, Y):
        grad_CE = self.cross_entropy_backward(AL, Y) * self.recon_loss_weight
        logvar = self.layers[self.bottleneck_layer].logvar
        dLogvar = -(2 / logvar - 2 * logvar)
        mu = self.layers[self.bottleneck_layer].mu
        self.layers[self.bottleneck_layer].dMu = 2 * mu * self.KLD_beta
        self.layers[self.bottleneck_layer].dLogvar = dLogvar
        return grad_CE

    def squared_means_KLD(self, AL, Y):
        SM = self.squared_means(AL, Y) * self.recon_loss_weight
        logvar = self.layers[self.bottleneck_layer].logvar
        mu = self.layers[self.bottleneck_layer].mu
        KL = -xp.sum(1 + xp.log(logvar**2) - mu**2 - logvar**2) * self.KLD_beta
        SM[0][0] += KL 
        return SM

    def squared_means_KLD_backward(self, AL, Y):
        grad_SM = self.squared_means_backward(AL, Y) * self.recon_loss_weight
        logvar = self.layers[self.bottleneck_layer].logvar
        dLogvar = -(2 / logvar - 2 * logvar) * self.KLD_beta
        mu = self.layers[self.bottleneck_layer].mu
        self.layers[self.bottleneck_layer].dMu = 2 * mu * self.KLD_beta
        self.layers[self.bottleneck_layer].dLogvar = dLogvar
        return grad_SM


    def compile(self, loss, threshold=0.5, recon_loss_weight=1., KLD_beta=0.7):
        if loss not in ["cross_entropy", "squared_means", "categorical_cross_entropy", "cross_entropy_KLD", "squared_means_KLD"]:
            raise Exception(f"invalid value: loss must be either 'cross_entropy', 'categorical_cross_entropy', 'cross_entropy_KLD', 'squared_means_KLD' or 'squared_means'. (is currently {loss})")
        self.loss = loss
        self.recon_loss_weight = recon_loss_weight
        self.threshold = threshold
        self.is_train = False
        self.KLD_beta = KLD_beta
        for func in [self.squared_means, self.cross_entropy, self.categorical_cross_entropy, self.cross_entropy_KLD, self.squared_means_KLD]:
            if func.__name__ == loss:
                self.loss_forward = func
        for func in [self.squared_means_backward, self.cross_entropy_backward, self.categorical_cross_entropy_backward, self.cross_entropy_KLD_backward, self.squared_means_KLD_backward]:
            if func.__name__[:-9] == loss:
                self.loss_backward = func
        self._is_compiled = True

    def compute_cost(self, AL, Y, m=0):
        if m == 0:
            m = AL.shape[1]
        costs = self.loss_forward(AL, Y)
        regularization_costs = 0
        for l in range(1, len(self.layers)):
            regularization_costs += self.layers[l].regularization_cost(m)
        cost = xp.sum(costs)/m + regularization_costs
        if self.use_cuda:
            cost = cost.item()
        return cost

    def train(self, X, Y, num_epochs, mini_batch_size=0, print_frequency=100):
        print_ind = max(num_epochs // print_frequency, 1)
        L = len(self.layers)
        if mini_batch_size == 0:
            mini_batch_size = Y.shape[1]
        costs = []
        for i in range(num_epochs):
            Ji = 0
            mini_batches = self.random_mini_batches(X, Y, mini_batch_size, i)
            for k in range(len(mini_batches)):
                #forward propagation
                Al = mini_batches[k][0]
                Yk = mini_batches[k][1]
                for l in range(1, L):
                    Al = self.layers[l].forward_propagation(Al, True)
                    #print(xp.isnan(Al).any())
                #backward propagation
                dAl = self.loss_backward(Al, Yk)
                for l in reversed(range(1, L)):
                    dAl = self.layers[l].backward_propagation(dAl)
                    #update parameters
                    self.layers[l].update_parameters(i+1)
                Ji += self.compute_cost(Al, Yk, Y.shape[-1])
            #record progress
            if i >= 0 and i % print_ind == 0:
                #J = self.compute_cost(Al, Y)
                #if i % 10 == 0:
                    #gen = cp.random.normal(0.5, 0.5, (96,4))
                    #for l in range(4,6):
                        #gen = self.layers[l].forward_propagation(gen, False)
                    #gen = cp.asnumpy(gen).T * 255.
                    #gen = np.clip(gen, 0, 255)
                    #for ii in range(4):
                        #plt.imsave(f"3gen {ii} epoch {i}.jpg", gen[ii].reshape(28,28), cmap = matplotlib.cm.binary)
                costs.append(Ji)
                inject_string = ""
                if self.inject_str_func != None:
                    inject_string = self.inject_str_func(self, X, Y, Al)
                print(f"Iteration: {i}, cost: {Ji}" + inject_string)
        return costs

    def predict(self, X):
        Al = X
        for l in range(1, len(self.layers)):
                Al = self.layers[l].forward_propagation(Al, False)
        if self.loss_forward == self.categorical_cross_entropy:
            return xp.where(Al==Al.max(axis=0),1,0)
        return xp.where(Al > self.threshold, 1, 0)

    def forward_propagation(self, X):
        Al = X
        for l in range(1, len(self.layers)):
                Al = self.layers[l].forward_propagation(Al, False)
        if self.loss == self.categorical_cross_entropy:
            return xp.where(Al==Al.max(axis=0),1,0)
        return Al

    def save_weights(self, path, is_cupy):
        for l in range(1, len(self.layers)):
            self.layers[l].save_weights(path, "Layer{i}".format(i = l), is_cupy)

    def load_weights(self, path):
        for l in range(1, len(self.layers)):
            self.layers[l].load_weights(path, l)

    def check_backward_propagation(self, X, Y, epsilon=1e-7):
        L = len(self.layers)
        #forward and backward prop to update dw and db in layers
        Al = X
        for l in range(1, L):
            Al = self.layers[l].forward_propagation(Al, False)
        dAl = self.loss_backward(Al, Y)
        for l in reversed(range(1, L)):
            dAl = self.layers[l].backward_propagation(dAl)
        #make return variables
        avg_diff = 0
        total_check = True
        problem_layer = L
        #calc the approx grad for each layer
        for l in reversed(range(1, L)):
            params_vec = self.layers[l].params_to_vec()
            grad_vec = self.layers[l].gradients_to_vec()
            n = len(params_vec)
            approx = xp.zeros((n,), dtype=float)
            for i in range(n):
                v = xp.copy(params_vec)
                v[i] += epsilon
                self.layers[l].vec_to_params(v)
                Al = X
                for layer in range(1, L):
                    Al = self.layers[layer].forward_propagation(Al, False)
                j_plus = self.compute_cost(Al, Y)
                v[i] -= 2*epsilon 
                self.layers[l].vec_to_params(v)
                Al = X
                for layer in range(1, L):
                    Al = self.layers[layer].forward_propagation(Al, False)
                j_minus = self.compute_cost(Al, Y) 
                approx[i] = j_plus-j_minus
            approx /= (2*epsilon) 
            diff = (xp.linalg.norm(grad_vec-approx))/((xp.linalg.norm(grad_vec))+(xp.linalg.norm(approx)))
            check = (diff < epsilon)
            self.layers[l].vec_to_params(params_vec)
            print(diff)
            avg_diff += diff 
            if not check:
                total_check = False
                problem_layer = l
        avg_diff /= (L - 1)
        return total_check, avg_diff, problem_layer

    @staticmethod
    def to_one_hot(num_categories, Y):
        m = Y.shape[0]
        Y = Y.reshape(1, m)
        Y_new = xp.eye(num_categories)[Y.astype('int32')]
        Y_new = Y_new.T
        Y_new = Y_new.reshape(num_categories, m)
        return Y_new

    def confusion_matrix(self, X, Y):
        '''prediction = self.predict(X)
        prediction_index = xp.argmax(prediction, axis=0)
        Y_index = xp.argmax(Y, axis=0)
        right = xp.sum(prediction_index == Y_index)
        print("accuracy: ",str(right/len(Y[0])))
        print(confusion_matrix(prediction_index, Y_index))'''
        prediction = self.predict(X)
        Y_np = Y
        if self.use_cuda:
            prediction = cp.asnumpy(prediction)
            Y_np = cp.asnumpy(Y)
        prediction_index = np.argmax(prediction, axis=0)
        Y_index = np.argmax(Y_np, axis=0)
        right = xp.sum(xp.array(prediction_index) == xp.array(Y_index))
        print("accuracy: ",str(right/len(Y[0])))
        print(confusion_matrix(prediction_index, Y_index))

    @staticmethod
    def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
        xp.random.seed(seed)
        m = Y.shape[1]
        permutation = list(xp.random.permutation(m))
        shuffled_X = xp.take(X, permutation, axis=-1)
        shuffled_Y = Y[:, permutation].reshape((-1,m))
        num_complete_minibatches = math.floor(m/mini_batch_size)
        mini_batches = []
        for k in range(num_complete_minibatches if m%mini_batch_size == 0 else num_complete_minibatches+1):
            mini_batch_X = xp.array(shuffled_X[..., mini_batch_size*k : (k+1) * mini_batch_size])
            mini_batch_Y = xp.array(shuffled_Y[:, mini_batch_size*k : (k+1) * mini_batch_size])
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches

class DLLayer:
    def __init__ (self, name, num_units, input_shape: tuple, activation="relu", 
                  W_initialization="random", alpha=0.01, optimization=None, regularization=None, keep_prob=0.6, samples_per_dim=1):
        if activation not in ["sigmoid", "tanh", "relu", "leaky_relu", "softmax", "trim_sigmoid", "trim_tanh", "trim_softmax", "vae_bottleneck"]: 
            raise Exception(f"invalid value: activation must be either 'sigmoid', 'trim_sigmoid', 'tanh', 'trim_tanh', 'relu', 'leaky_relu', 'softmax', 'trim_softmax' or , 'vae_bottleneck'. (is currently {activation})")
        if W_initialization not in ["random", "zeros", "He", "Xavier"] and W_initialization[-3:] != ".h5":
            print(f"Warning: W_initialization is currently {W_initialization}, which may be incorrect. it should be either 'random', 'He', 'Xavier', 'zeros' or a path to a .h5 file.")
        if not (optimization is None) and optimization not in ["adaptive", "momentum", "adam", "rmsprop"]:
            raise Exception(f"invalid value: optimization must be either None, 'adaptive', 'adam', 'rmsprop' or 'momentum'. (is currently {optimization})")
        if activation == "vae_bottleneck" and num_units % 2 != 0:
            raise Exception(f"invalid value: 'vae_bottleneck' layers can only have an even input shape n. (is currently {input_shape})")
        self.name = name
        self._samples_per_dim = samples_per_dim
        self._num_units = num_units
        self._activation = activation
        self.W_initialization = W_initialization
        self.alpha = alpha
        self._optimization = optimization
        self.random_scale = 0.01
        self.input_shape = input_shape
        self.regularization = regularization
        self.L2_lambda = 0.
        if activation == "leaky_relu":
            self.leaky_relu_d = 0.3
        if activation[:4] == "trim":
            self.activation_trim = 1e-10
        for func in [self._sigmoid, self._trim_sigmoid, self._tanh, self._trim_tanh, self._relu, self._leaky_relu,
                    self._softmax, self._trim_softmax, self._vae_bottleneck]:
            if func.__name__[1:] == activation:
                self.activation_forward = func
                break
        for func in [self._sigmoid_backward, self._tanh_backward, self._relu_backward, 
                     self._leaky_relu_backward, self._trim_tanh_backward, self._trim_sigmoid_backward,
                    self._softmax_backward, self._trim_softmax_backward, self._vae_bottleneck_backward]:
            if func.__name__[1:-9] == activation:
                self.activation_backward = func
                break
        self.init_optimization_params()
        if regularization == "L2":
            self.L2_lambda = 0.6
        elif regularization == "dropout":
            self.dropout_keep_prob = keep_prob
        self.init_weights(W_initialization)

    def init_weights(self, W_initialization):
        self.b = xp.zeros((self._num_units,1), dtype=float)
        if W_initialization == "zeros":
            self.W = xp.zeros((self._num_units, *(self.input_shape)), dtype=float)
        elif W_initialization == "He":
            self.W = xp.random.randn(self._num_units, *(self.input_shape)) * xp.sqrt(2/self.input_shape[0])
        elif W_initialization == "Xavier":
            self.W = xp.random.randn(self._num_units, *(self.input_shape)) * xp.sqrt(1/self.input_shape[0])
        elif W_initialization == "random":
            self.W = xp.random.randn(self._num_units, *(self.input_shape)) * self.random_scale
        else: 
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = xp.array(hf['W'][:])
                    self.b = xp.array(hf['b'][:])
            except (FileNotFoundError):
                raise NotImplementedError("Unrecognized initialization:", W_initialization)

    def init_optimization_params(self):
        if self._optimization == "adaptive":
            self._adaptive_alpha_b = xp.full((self._num_units, 1), self.alpha)
            self._adaptive_alpha_W = xp.full((self._num_units, *(self.input_shape)),self.alpha)
            self.adaptive_cont = 1.1
            self.adaptive_switch = -0.5
        elif self._optimization == "momentum":
            self._v_dW = xp.zeros((self._num_units, *(self.input_shape)), dtype=float)
            self._v_db = xp.zeros((self._num_units,1), dtype=float)
            self.momentum_beta = 0.09
        elif self._optimization == "adam":
            self._adam_v_dW = xp.zeros((self._num_units, *(self.input_shape)), dtype=float)
            self._adam_v_db = xp.zeros((self._num_units,1), dtype=float)
            self._adam_s_dW = xp.zeros((self._num_units, *(self.input_shape)), dtype=float)
            self._adam_s_db = xp.zeros((self._num_units,1), dtype=float)
            self.adam_beta1 = 0.09
            self.adam_beta2 = 0.0999
            self.adam_epsilon = 1e-8
        elif self._optimization == "rmsprop":
            self._rmsprop_v_dW = xp.zeros((self._num_units, *(self.input_shape)), dtype=float)
            self._rmsprop_v_db = xp.zeros((self._num_units,1), dtype=float)
            self.rmsprop_beta = 0.0999
            self.rmsprop_epsilon = 1e-8

    def __str__(self):
        s = self.name + " Layer:\n"
        s += "\tnum_units: " + str(self._num_units) + "\n"
        s += "\tactivation: " + self._activation + "\n"
        if self._activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d)+"\n"
        s += "\tinput_shape: " + str(self.input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"
        #optimization
        if self._optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self.adaptive_cont)+"\n"
            s += "\t\t\tswitch: " + str(self.adaptive_switch)+"\n"
        elif self._optimization == "momentum":
            s += "\t\tmomentum parameters:\n"
            s += "\t\t\tbeta: " + str(self.momentum_beta)+"\n"
        #regularization
        s += "\tregularization: "
        s += self.regularization if self.regularization != None else "None"
        # parameters
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape)+"\n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()
        return s
        
    def get_output_shape(self):
        if self._activation == "vae_bottleneck":
            return (self._num_units * self._samples_per_dim // 2,)
        return (self._num_units,)

    def _sigmoid(self, Z):
        return 1/(1+xp.exp(-Z)) 

    def _tanh(self, Z):
        return xp.tanh(Z)

    def _relu(self, Z):
        return xp.maximum(0, Z)

    def _leaky_relu(self, Z):
        return xp.where(Z > 0, Z, Z * self.leaky_relu_d)

    def _softmax(self, Z):
        eZ = xp.exp(Z)
        return eZ/xp.sum(eZ, 0)

    def _trim_softmax(self, Z):
        Z -= xp.max(Z, axis=0)
        eZ = xp.exp(Z)
        A = eZ/xp.sum(eZ, axis=0)
        return A

    def _trim_sigmoid(self,Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1/(1+xp.exp(-Z))
            except FloatingPointError:
                Z = xp.where(Z < -100, -100,Z)
                A = A = 1/(1+xp.exp(-Z))
        TRIM = self.activation_trim
        if TRIM > 0:
            A = xp.where(A < TRIM,TRIM,A)
            A = xp.where(A > 1-TRIM,1-TRIM, A)
        return A

    def _trim_tanh(self,Z):
        A = xp.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = xp.where(A < -1+TRIM,TRIM,A)
            A = xp.where(A > 1-TRIM,1-TRIM, A)
        return A

    def _vae_bottleneck(self,Z):
        n = Z.shape[0]
        self.mu = Z[:n//2, :]
        self.logvar = Z[n//2:, :]
        #reparameterization trick
        self._vae_epsilon = xp.random.standard_normal(size=(n//2, Z.shape[1]))
        rand_sample = self.mu + xp.log(self.logvar**2) * xp.random.standard_normal(size=(n//2, Z.shape[1]))
        for i in range(1, self._samples_per_dim):
            extra_sample = self.mu + xp.log(self.logvar**2) * xp.random.standard_normal(size=(n//2, Z.shape[1]))
            rand_sample = xp.concatenate((rand_sample, extra_sample), axis=0)
        return rand_sample


    def forward_propagation(self, A_prev, is_train):
        self._A_prev = self.forward_dropout(A_prev, is_train)
        self._Z = xp.dot(self.W, self._A_prev) + self.b
        A = self.activation_forward(self._Z)
        return A

    def forward_dropout(self, A_prev, is_train):
        A_prev_copy = xp.array(A_prev, copy=True)
        if is_train == True and self.regularization == "dropout":
            self._D = xp.random.rand(1, A_prev.shape[0]) < self.dropout_keep_prob
            self._D = self._D.T
            A_prev_copy *= self._D
            A_prev_copy /= self.dropout_keep_prob
        return A_prev_copy

    def _sigmoid_backward(self, dA):
        A = self._sigmoid(self._Z)
        dZ = dA * A * (1 - A)
        return dZ
    
    def _trim_sigmoid_backward(self,dA):
        A = self._trim_sigmoid(self._Z)
        dZ = dA * A * (1-A)
        return dZ

    def _trim_tanh_backward(self,dA):
        A = self._trim_tanh(self._Z)
        dZ = dA * (1-A**2)
        return dZ

    def _tanh_backward(self, dA):
        dZ = (1 - xp.tanh(self._Z)**2) * dA
        return dZ

    def _relu_backward(self,dA):
        dZ = xp.where(self._Z <= 0, 0, dA)
        return dZ

    def _leaky_relu_backward(self, dA):
        dZ = xp.where(self._Z <= 0, dA * self.leaky_relu_d, dA)
        return dZ
    
    def _softmax_backward(self, dZ):
        return dZ

    def _trim_softmax_backward(self, dZ):
        return dZ

    def _vae_bottleneck_backward(self, dA):
        dA = dA.reshape(dA.shape[0] // self._samples_per_dim, self._samples_per_dim, dA.shape[1])
        dA = xp.mean(dA, axis=1)
        dZ = xp.concatenate((dA + self.dMu, self._vae_epsilon * dA + self.dLogvar), axis=0)
        return dZ

    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        m = dZ.shape[1]
        self.db = xp.sum(dZ , axis=1, keepdims=True)/m
        self.dW = ((dZ @ (self._A_prev.T)) + self.L2_lambda * self.W)/m
        dA_Prev = self.W.T @ dZ
        dA_Prev = self.backward_dropout(dA_Prev)
        return dA_Prev

    def backward_dropout(self, dA_prev):
        if self.regularization == "dropout":
            dA_prev *= self._D
            dA_prev /= self.dropout_keep_prob
        return dA_prev

    def regularization_cost(self, m):
        if self.regularization != "L2":
            return 0 
        W_square_sum = xp.sum(xp.square(self.W))
        return self.L2_lambda * W_square_sum / (2 * m)

    def update_parameters(self, t=1):
        if self._optimization is None:
            self.W -= self.dW * self.alpha
            self.b -= self.db * self.alpha
        elif self._optimization == "adaptive":
            self._adaptive_alpha_W *= xp.where(self._adaptive_alpha_W * self.dW > 0, self.adaptive_cont, self.adaptive_switch)
            self._adaptive_alpha_b *= xp.where(self._adaptive_alpha_b * self.db > 0, self.adaptive_cont, self.adaptive_switch)
            self.W -= self._adaptive_alpha_W
            self.b -= self._adaptive_alpha_b
        elif self._optimization == "momentum":
            self._v_dW = self.momentum_beta * self._v_dW + (1-self.momentum_beta) * self.dW
            self._v_db = self.momentum_beta * self._v_db + (1-self.momentum_beta) * self.db
            self.W -= self.alpha * self._v_dW
            self.b -= self.alpha * self._v_db
        elif self._optimization == "adam":
            self._adam_v_dW = (self.adam_beta1 * self._adam_v_dW + (1-self.adam_beta1) * self.dW)/(1-self.adam_beta1**t)
            self._adam_v_db = (self.adam_beta1 * self._adam_v_db + (1-self.adam_beta1) * self.db)/(1-self.adam_beta1**t)
            self._adam_s_dW = (self.adam_beta2 * self._adam_s_dW + (1-self.adam_beta2) * self.dW**2)/(1-self.adam_beta2**t)
            self._adam_s_db = (self.adam_beta2 * self._adam_s_db + (1-self.adam_beta2) * self.db**2)/(1-self.adam_beta2**t)
            self.W -= self.alpha * self._adam_v_dW / xp.sqrt(self._adam_s_dW + self.adam_epsilon)
            self.b -= self.alpha * self._adam_v_db / xp.sqrt(self._adam_s_db + self.adam_epsilon)
        elif self._optimization == "rmsprop":
            self._rmsprop_v_dW = self.rmsprop_beta * self._rmsprop_v_dW + (1-self.rmsprop_beta) * self.dW**2
            self._rmsprop_v_db = self.rmsprop_beta * self._rmsprop_v_db + (1-self.rmsprop_beta) * self.db**2
            self.W -= self.alpha * self.dW / xp.sqrt(self._rmsprop_v_dW+self.rmsprop_epsilon)
            self.b -= self.alpha * self.db /xp.sqrt(self._rmsprop_v_db+self.rmsprop_epsilon)

    def save_weights(self, path, file_name, is_cupy=False):
        if not os.path.exists(path):
            os.makedirs(path)
        with h5py.File(path+"/"+file_name+'.h5', 'w') as hf:
            if is_cupy:
                hf.create_dataset("W", data=xp.asnumpy(self.W))
                hf.create_dataset("b", data=xp.asnumpy(self.b))
            else:
                hf.create_dataset("W", data=self.W)
                hf.create_dataset("b", data=self.b)

    def params_to_vec(self):
        return xp.concatenate((xp.reshape(self.W,(-1,)),xp.reshape(self.b, (-1,))), axis=0)

    def vec_to_params(self, vec):
        self.W = vec[0:self.W.size].reshape(self.W.shape)
        self.b = vec[self.W.size:].reshape(self.b.shape)

    def gradients_to_vec(self):
        return xp.concatenate((xp.reshape(self.dW,(-1,)),xp.reshape(self.db, (-1,))), axis=0)

    def load_weights(self, path, layer_index):
        try:
            with h5py.File(path+"/Layer"+str(layer_index)+".h5", 'r') as hf:
                self.W = xp.array(hf['W'][:])
                self.b = xp.array(hf['b'][:])
        except (FileNotFoundError):
            raise FileNotFoundError(f"Couldn't find layer {layer_index} in {path}.")

class DLConvLayer(DLLayer):

    def __init__(self, name, num_filters, input_shape, activation="relu", W_initialization="random", alpha=0.01, filter_size=(3,3), strides=(1,1), 
                 padding="same", optimization=None, regularization=None):
        if padding not in ["same", "valid"] and type(padding) != tuple:
            raise Exception(f"invalid value: padding must be either a tuple, 'same' or 'valid'. (is currently {padding})")
        self.filter_size = filter_size
        super().__init__(name, num_filters, input_shape, activation, W_initialization, alpha, optimization, regularization)
        self.padding = padding
        if padding == "valid":
            self.padding = (0,0)
        elif padding == "same":
            py = int((strides[0]*input_shape[1]-strides[0]-input_shape[1]+filter_size[0]+1)/2)
            px = int((strides[1]*input_shape[2]-strides[1]-input_shape[2]+filter_size[1]+1)/2)
            self.padding = (py, px)
        
        self.h_out = int((input_shape[1] + self.padding[0]*2 - filter_size[0])/strides[0]+1)
        self.w_out = int((input_shape[2] + self.padding[1]*2 - filter_size[1])/strides[1]+1)
        self.strides = strides

    def get_output_shape(self):
        return (self._num_units, self.h_out, self.w_out) 

    def init_weights(self, W_initialization):
        self.b = xp.zeros((self._num_units,1), dtype=float)
        if W_initialization == "zeros":
            self.W = xp.zeros((self._num_units, self.input_shape[0], *(self.filter_size)), dtype=float)
        elif W_initialization == "He":
            self.W = xp.random.randn(self._num_units, self.input_shape[0], *(self.filter_size)) * xp.sqrt(2/(self.filter_size[0] * self.filter_size[1] * self.input_shape[0]))
        elif W_initialization == "Xavier":
            fan_in = self.filter_size[0] * self.filter_size[1] * self.input_shape[0]
            fan_out = self._num_units * self.filter_size[0] * self.filter_size[1]
            self.W = xp.random.randn(self._num_units, self.input_shape[0], *(self.filter_size)) * xp.sqrt(2 / (fan_in + fan_out))
        elif W_initialization == "random":
            self.W = xp.random.randn(self._num_units, self.input_shape[0], *(self.filter_size)) * self.random_scale
        else: 
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = xp.array(hf['W'][:])
                    self.b = xp.array(hf['b'][:])
            except (FileNotFoundError):
                raise NotImplementedError("Unrecognized initialization:", W_initialization)
        
    def init_optimization_params(self):
        if self._optimization == "adaptive":
            self._adaptive_alpha_b = xp.full((self._num_units, 1), self.alpha)
            self._adaptive_alpha_W = xp.full((self._num_units, self.input_shape[0], *(self.filter_size)),self.alpha)
            self.adaptive_cont = 1.1
            self.adaptive_switch = -0.5
        elif self._optimization == "momentum":
            self._v_dW = xp.zeros((self._num_units, self.input_shape[0], *(self.filter_size)), dtype=float)
            self._v_db = xp.zeros((self._num_units,1), dtype=float)
            self.momentum_beta = 0.09
        elif self._optimization == "adam":
            self._adam_v_dW = xp.zeros((self._num_units, self.input_shape[0], *(self.filter_size)), dtype=float)
            self._adam_v_db = xp.zeros((self._num_units,1), dtype=float)
            self._adam_s_dW = xp.zeros((self._num_units, self.input_shape[0], *(self.filter_size)), dtype=float)
            self._adam_s_db = xp.zeros((self._num_units,1), dtype=float)
            self.adam_beta1 = 0.09
            self.adam_beta2 = 0.0999
            self.adam_epsilon = 1e-8
        elif self._optimization == "rmsprop":
            self._rmsprop_v_dW = xp.zeros((self._num_units, self.input_shape[0], *(self.filter_size)), dtype=float)
            self._rmsprop_v_db = xp.zeros((self._num_units,1), dtype=float)
            self.rmsprop_beta = 0.0999
            self.rmsprop_epsilon = 1e-8

    def __str__(self):
        s = "Convolutional " + super(DLConvLayer, self).__str__()
        s += "\tConvolutional parameters:\n"
        s += f"\t\tfilter size: {self.filter_size}\n"
        s += f"\t\tstrides: {self.strides}\n"
        s += f"\t\tpadding: {self.padding}\n"
        s += f"\t\toutput shape: {(self._num_units, self.h_out, self.w_out)}\n"
        return s

    @staticmethod
    def im2col_indices(A, filter_height=3, filter_width=3, padding=(1,1),stride=(1,1)):
        """ An implementation of im2col based on some fancy indexing """
        # Zero-pad the input
        A_padded = np.pad(A, ((0, 0), (0, 0), (padding[0], padding[1]), (padding[0], padding[1])), mode='constant')

        k, i, j = DLConvLayer.get_im2col_indices(A.shape, filter_height, filter_width, padding, stride)

        cols = A_padded[:, k, i, j]
        C = A.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * C, -1)
        return cols

    @staticmethod
    def get_im2col_indices(A_shape, filter_height=3, filter_width=3, padding=(1,1),stride=(1,1)):
        # First figure out what the size of the output should be
        m, C, H, W = A_shape
        #assert (H + 2 * padding[0] - filter_height) % stride[0] == 0
        #assert (W + 2 * padding[1] - filter_width) % stride[1] == 0
        out_height = int((H + 2 * padding[0] - filter_height) / stride[0]) + 1
        out_width = int((W + 2 * padding[1] - filter_width) / stride[1]) + 1

        i0 = xp.repeat(xp.arange(filter_height), filter_width)
        i0 = xp.tile(i0, C)
        i1 = stride[0] * xp.repeat(xp.arange(out_height), out_width)
        j0 = xp.tile(xp.arange(filter_width), filter_height * C)
        j1 = stride[1] * xp.tile(xp.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = xp.repeat(xp.arange(C), filter_height * filter_width).reshape(-1, 1)

        return (k, i, j)

    def col2im_indices(cols, A_shape, filter_height=3, filter_width=3, padding=(1,1),stride=(1,1)):
        """ An implementation of col2im based on fancy indexing and np.add.at """
        m, C, H, W = A_shape
        H_padded, W_padded = H + 2 * padding[0], W + 2 * padding[1]
        A_padded = xp.zeros((m, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = DLConvLayer.get_im2col_indices(A_shape, filter_height, filter_width, padding, stride)
        cols_reshaped = cols.reshape(C * filter_height * filter_width, -1, m)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        if xp == np:
            xp.add.at(A_padded, (slice(None), k, i, j), cols_reshaped)
        else:
            cpx.scatter_add(A_padded, (slice(None), k, i, j), cols_reshaped)
        if padding[0] == 0 and padding[1] == 0:
            return A_padded
        if padding[0] == 0:
            return A_padded[:, :, :, padding[1]:-padding[1]]
        if padding[1] == 0:
            return A_padded[:, :, padding[0]:-padding[0], :]
        return A_padded[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

    def forward_propagation(self, A_prev, is_train):
        A_prev = xp.transpose(A_prev, (3, 0, 1, 2))
        A_prev = DLConvLayer.im2col_indices(A_prev, self.filter_size[0], self.filter_size[1], self.padding, self.strides)
        self._A_prev = self.forward_dropout(A_prev, is_train)
        W_temp = self.W
        self.W = self.W.reshape(self._num_units, -1)
        self._Z = xp.dot(self.W, self._A_prev) + self.b
        self._Z = self._Z.reshape(self._num_units, self.h_out, self.w_out, -1)
        self.W = W_temp
        A = self.activation_forward(self._Z)
        return A

    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        m = dZ.shape[-1]
        dZ = dZ.reshape(self._num_units, -1)
        W_temp = self.W
        self.W = self.W.reshape(self._num_units, -1)
        self.db = xp.sum(dZ , axis=1, keepdims=True)/m
        self.dW = ((dZ @ (self._A_prev.T)) + self.L2_lambda * self.W)/m
        dA_Prev = self.W.T @ dZ
        dA_Prev = self.backward_dropout(dA_Prev)
        self.W = W_temp
        self.dW = self.dW.reshape(*(self.W.shape))
        A_prev_shape = (m,*self.input_shape)
        dA_Prev = DLConvLayer.col2im_indices(dA_Prev, A_prev_shape, self.filter_size[0], self.filter_size[1], self.padding, self.strides)
        # transpose dA-prev from (m,C,H,W) to (C,H,W,m)
        dA_Prev = dA_Prev.transpose(1,2,3,0)
        return dA_Prev

class DLMaxPoolingLayer:

    def __init__(self, name, input_shape, filter_size=(3,3), strides=(1,1)):
        self.name = name
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.strides = strides
        self.h_out = int((input_shape[1] - filter_size[0])/strides[0]+1)
        self.w_out = int((input_shape[2] - filter_size[1])/strides[1]+1)

    def __str__(self):
        s = f"Maxpooling {self.name} Layer:\n"
        s += f"\tinput_shape: {self.input_shape}\n"
        s += "\tMaxpooling parameters:\n"
        s += f"\t\tfilter size: {self.filter_size}\n"
        s += f"\t\tstrides: {self.strides}\n"
        # number of output channels == number of input channels
        s += f"\t\toutput shape: {(self.input_shape[0], self.h_out, self.w_out)}\n"
        return s

    def forward_propagation(self, A_prev, is_train=False):
        # first transpose A_prev from (C,H,W,m) to (m,C,H,W)
        A_prev = A_prev.transpose(3, 0, 1, 2)
        m,C,H,W = A_prev.shape
        A_prev = A_prev.reshape(m*C,1,H,W)
        self._A_prev = DLConvLayer.im2col_indices(A_prev, self.filter_size[0], self.filter_size[1], padding = (0,0), stride = self.strides)
        self.max_indexes = xp.argmax(self._A_prev,axis=0)
        Z = self._A_prev[self.max_indexes,range(self.max_indexes.size)]
        Z = Z.reshape(self.h_out,self.w_out,m,C).transpose(3,0,1,2)
        return Z

    def backward_propagation(self,dZ):
        dA_prev = np.zeros_like(self._A_prev)
        # transpose dZ from C,h,W,C to H,W,m,c and flatten it
        # Then, insert dZ values to dA_prev in the places of the max indexes
        dZ_flat = dZ.transpose(1,2,3,0).ravel()
        dA_prev[self.max_indexes,range(self.max_indexes.size)] = dZ_flat
        # get the original prev_A structure from col2im
        m = dZ.shape[-1]
        C,H,W = self.input_shape
        shape = (m*C,1,H,W)
        dA_prev = DLConvLayer.col2im_indices(dA_prev, shape, self.filter_size[0], self.filter_size[1], padding=(0,0),stride=self.strides)
        dA_prev = dA_prev.reshape(m,C,H,W).transpose(1,2,3,0)
        return dA_prev

    def get_output_shape(self):
        return (self.input_shape[0], self.h_out, self.w_out) 

    def update_parameters(self, t=1):
        #for compatibility with DLModel
        return

    def regularization_cost(self, m=1):
        #for compatibility with DLModel
        return 0

    def save_weights(self, path, file_name, is_cupy=False):
        #for compatibility with DLModel
        return  

    def load_weights(self, path, layer_index):
        #for compatibility with DLModel
        return

class DLFlattenLayer:

    def __init__(self, name, input_shape):
        self.input_shape = input_shape
        self.name = name

    def __str__(self):
        s = f"Flatten {self.name} Layer:\n"
        s += f"\tinput_shape: {self.input_shape}\n"
        return s

    def forward_propagation(self, prev_A, is_train=False):
        m = prev_A.shape[-1]
        # for TF competability transpose to (H,W,C, m) before flattening
        A = prev_A.transpose(1,2,0,3).reshape(-1,m)
        return A

    def backward_propagation(self,dA):
        m = dA.shape[-1]
        # reshape back to (H,W,C, m) before transposing to (C,H,W, m)
        C,H,W = self.input_shape
        dA_prev = dA.reshape(H,W,C,m).transpose(2,0,1,3)
        return dA_prev

    def get_output_shape(self):
        return (self.input_shape[0] * self.input_shape[1] * self.input_shape[2],)

    def update_parameters(self, t=1):
        #for compatibility with DLModel
        return

    def regularization_cost(self, m=1):
        #for compatibility with DLModel
        return 0

    def save_weights(self, path, file_name, is_cupy=False):
        #for compatibility with DLModel
        return 

    def load_weights(self, path, layer_index):
        #for compatibility with DLModel
        return

class DLDeflattenLayer:

    def __init__(self, name, input_shape, output_shape):
        self.input_shape = input_shape
        self.name = name
        self.output_shape = output_shape

    def __str__(self):
        s = f"Deflatten {self.name} Layer:\n"
        s += f"\tinput_shape: {self.input_shape}\n"
        return s

    def backward_propagation(self, prev_A):
        m = prev_A.shape[-1]
        # for TF competability transpose to (H,W,C, m) before flattening
        A = prev_A.transpose(1,2,0,3).reshape(-1,m)
        return A

    def forward_propagation(self,dA, is_train=False):
        m = dA.shape[-1]
        # reshape back to (H,W,C, m) before transposing to (C,H,W, m)
        C,H,W = self.output_shape
        dA_prev = dA.reshape(H,W,C,m).transpose(2,0,1,3)
        return dA_prev

    def get_output_shape(self):
        return self.output_shape

    def update_parameters(self, t=1):
        #for compatibility with DLModel
        return

    def regularization_cost(self, m=1):
        #for compatibility with DLModel
        return 0

    def save_weights(self, path, file_name, is_cupy=False):
        #for compatibility with DLModel
        return 

    def load_weights(self, path, layer_index):
        #for compatibility with DLModel
        return

class DLUpsampleLayer:

    def __init__(self, name, input_shape, scale=2):
        self.name = name
        self.input_shape = input_shape
        self.scale = scale

    def __str__(self):
        s = f"Upsample {self.name} Layer:\n"
        s += f"\tinput_shape: {self.input_shape}\n"
        s += f"\tscale: {self.scale}\n"
        return s

    def forward_propagation(self, A_prev, is_train=False):
        A_prev = A_prev.transpose(0, 1, 2, 3)
        if xp == np:
            Z = ndimage.zoom(A_prev, (1, self.scale, self.scale, 1))
        else:
            Z = cpx.scipy.ndimage.zoom(A_prev, (1, self.scale, self.scale, 1))
        return Z

    def backward_propagation(self,dZ):
        dZ = dZ.transpose(0, 1, 2, 3)
        if xp == np:
            dA_prev = ndimage.zoom(dZ, (1, 1/self.scale, 1/self.scale, 1))
        else:
            dA_prev = cpx.scipy.ndimage.zoom(dZ, (1, 1/self.scale, 1/self.scale, 1))
        return dA_prev

    def get_output_shape(self):
        return (self.input_shape[0], int(self.input_shape[1]*self.scale), int(self.input_shape[2]*self.scale)) 

    def update_parameters(self, t=1):
        #for compatibility with DLModel
        return

    def regularization_cost(self, m=1):
        #for compatibility with DLModel
        return 0

    def save_weights(self, path, file_name, is_cupy=False):
        #for compatibility with DLModel
        return  

    def load_weights(self, path, layer_index):
        #for compatibility with DLModel
        return

class DLPixelShuffleLayer:

    def __init__(self, name, input_shape):
        self.name = name
        if input_shape[0] % 4 != 0:
            raise Exception(f"invalid value: input shape must have a channel amount that is divisable by 4. (is currently {input_shape}")
        self.input_shape = input_shape
        self._slice_index = input_shape[0] // 4

    def __str__(self):
        s = f"Pixel Shuffle {self.name} Layer:\n"
        s += f"\tinput_shape: {self.input_shape}\n"
        return s

    def get_output_shape(self):
        return (self._slice_index, self.input_shape[1] * 2, self.input_shape[2] * 2)

    def forward_propagation(self, A_Prev, is_train=False):
        #no insert in cupy :(
        if xp != np:
            A_Prev = xp.asnumpy(A_Prev)
        #slice A_Prev by channels into 4 slices
        s1 = A_Prev[:self._slice_index]
        s2 = A_Prev[self._slice_index:self._slice_index*2]
        s3 = A_Prev[self._slice_index*2:self._slice_index*3]
        s4 = A_Prev[self._slice_index*3:self._slice_index*4]
        #combine the slices with each other
        combined_s1s2 = np.insert(s1, np.arange(s1.shape[2]), s2, axis=2)
        combined_s3s4 = np.insert(s3, np.arange(s1.shape[2]), s4, axis=2)
        Z = np.insert(combined_s1s2, np.arange(s1.shape[1]), combined_s3s4, axis=1)
        if xp != np:
            Z = xp.array(Z)
        return Z

    def backward_propagation(self, dZ):
        combined_s1s2 = dZ[:,::2,:,:]
        combined_s3s4 = dZ[:,1::2,:,:]
        s1 = combined_s1s2[:,:,::2,:]
        s2 = combined_s1s2[:,:,1::2,:]
        s3 = combined_s3s4[:,:,::2,:]
        s4 = combined_s3s4[:,:,1::2,:]
        dA_Prev = xp.concatenate((s4,s3,s2,s1), axis=0)
        return dA_Prev

    def update_parameters(self, t=1):
        #for compatibility with DLModel
        return

    def regularization_cost(self, m=1):
        #for compatibility with DLModel
        return 0

    def save_weights(self, path, file_name, is_cupy=False):
        #for compatibility with DLModel
        return 

    def load_weights(self, path, layer_index):
        #for compatibility with DLModel
        return

class DLGANModel:

    def __init__(self, name, generator, discriminator):
        self.name = name
        self.generator = generator
        self.discriminator = discriminator

    def __str__(self):
        s = f"{self.name} GAN model description:"
        s += f"\tgenerator: {self.generator}"
        s += f"\tdiscriminator: {self.discriminator}"
        return s

    def compile(self, loss="wasserstein"):
        if loss not in ["minimax", "wasserstein"]:
            raise Exception(f"invalid value: loss must be either 'minimax' or 'wasserstein'. (is currently {loss})")
        self.loss = loss
        self.avg_fake_score = 0
        self.avg_real_score = 0
        self.train_discriminator = self.train_non_minimax_discriminator
        if loss == "wasserstein":
            self.generator_loss_forward = self.generator_wasserstein_loss_forward
            self.generator_loss_backward = self.generator_wasserstein_loss_backward
            self.discriminator_loss_forward = self.discriminator_wasserstein_loss_forward
            self.discriminator_loss_backward = self.discriminator_wasserstein_loss_backward
        if loss == "minimax":
            self.generator_loss_forward = self.generator_minimax_loss_forward
            self.generator_loss_backward = self.generator_minimax_loss_backward
            self.train_discriminator = self.train_minimax_discriminator

    def generator_minimax_loss_forward(self, Gz, m=0):
        D_Gz = self.discriminator.forward_propagation(Gz)
        Y = xp.ones(D_Gz.shape)
        E_z = self.discriminator.compute_cost(D_Gz, Y, m)
        return E_z

    def generator_minimax_loss_backward(self, D_Gz):
        D_Gz = xp.where(D_Gz == 0, D_Gz+1e-10, D_Gz )
        D_Gz = xp.where(D_Gz == 1, D_Gz-1e-10, D_Gz )
        return -1/D_Gz

    def generator_wasserstein_loss_forward(self, Gz, m):
        D_Gz = self.discriminator.forward_propagation(Gz)
        return xp.sum(D_Gz)/m

    def generator_wasserstein_loss_backward(self, D_Gz):
        return -xp.ones(D_Gz.shape)

    def discriminator_wasserstein_loss_forward(self, AL, Y, m):
        a = xp.where(Y == -1, AL, 0)
        b = xp.where(Y == 1, AL, 0)
        self.avg_real_score = xp.mean(a)
        self.avg_fake_score = xp.mean(b)
        return xp.sum(AL*Y)/m

    def discriminator_wasserstein_loss_backward(self, AL, Y):
        return Y

    def _generate_noise(self, m):
        n = self.generator.layers[1].input_shape[0]
        return xp.random.normal(0, 160, (*self.generator.layers[1].input_shape, m))

    def generate(self, m=1):
        noise = self._generate_noise(m)
        out = self.generator.forward_propagation(noise)
        return out

    def train_generator(self, num_epochs, m, mini_batch_size=0):
        print_ind = max(num_epochs // 1, 1)
        X = self._generate_noise(m)
        Y = xp.zeros((1,m))
        L = len(self.generator.layers)
        if mini_batch_size == 0:
            mini_batch_size = m
        costs = []
        for i in range(num_epochs):
            Ji = 0
            mini_batches = self.generator.random_mini_batches(X, Y, mini_batch_size, i)
            for k in range(len(mini_batches)):
                #forward propagation
                Al = mini_batches[k][0]
                #Yk = mini_batches[k][1]
                for l in range(1, L):
                    Al = self.generator.layers[l].forward_propagation(Al, True)
                #backward propagation
                D_Gz = self.discriminator.forward_propagation(Al)
                #print(xp.mean(D_Gz))
                dAl = self.generator_loss_backward(D_Gz)
                #print(dAl)
                for l in reversed(range(1, len(self.discriminator.layers))):
                    dAl = self.discriminator.layers[l].backward_propagation(dAl)
                #print(dAl)
                for l in reversed(range(1, L)):
                    dAl = self.generator.layers[l].backward_propagation(dAl)
                    #update parameters
                    self.generator.layers[l].update_parameters(i+1)
                Ji += self.generator_loss_forward(Al, m)
            #record progress
            if i >= 0 and i % print_ind == 0:
                costs.append(Ji)
                print(self.report_generator_train_stats(i,Ji))
        return costs

    def report_generator_train_stats(self, i, Ji):
        return f"Generator: iteration: {i}, cost: {Ji}"

    def train_non_minimax_discriminator(self, X_real, num_epochs, mini_batch_size=0):
        print_ind = max(num_epochs // 1, 1)
        m = X_real.shape[-1]
        Gz = self.generate(m)
        X = xp.concatenate((X_real, Gz), axis=-1)
        Y_real = -xp.ones((1,m))
        Y_fake = xp.ones((1,m))
        Y = xp.concatenate((Y_real, Y_fake), axis=-1)
        L = len(self.discriminator.layers)
        if mini_batch_size == 0:
            mini_batch_size = m
        costs = []
        for i in range(num_epochs):
            Ji = 0
            mini_batches = self.discriminator.random_mini_batches(X, Y, mini_batch_size, i)
            for k in range(len(mini_batches)):
                #forward propagation
                Al = mini_batches[k][0]
                Yk = mini_batches[k][1]
                #print(Yk)
                #print(Al)
                for l in range(1, L):
                    Al = self.discriminator.layers[l].forward_propagation(Al, True)
                #backward propagation
                dAl = self.discriminator_loss_backward(Al, Yk)
                #print(dAl)
                for l in reversed(range(1, L)):
                    dAl = self.discriminator.layers[l].backward_propagation(dAl)
                    #update parameters
                    self.discriminator.layers[l].update_parameters(i+1)
                    if hasattr(self.discriminator.layers[l], "W"):
                        self.discriminator.layers[l].W = xp.clip(self.discriminator.layers[l].W, -0.01, 0.01)
                Ji += self.discriminator_loss_forward(Al, Yk, m)
            #record progress
            if i >= 0 and i % print_ind == 0:
                costs.append(Ji)
                print(self.report_discriminator_train_stats(i,Ji))
        return costs

    def report_discriminator_train_stats(self, i, Ji):
        return f"Discriminator: iteration: {i}, cost: {Ji}\naverage real score: {self.avg_real_score}, average fake score: {self.avg_fake_score}"

    def train_minimax_discriminator(self, X, num_epochs, mini_batch_size=0):
        m = X.shape[-1]
        Gz = self.generate(m)
        X_total = xp.concatenate((X, Gz), axis=-1)
        Y_ones = xp.ones((1,m))
        Y_zeros = xp.zeros((1,m))
        Y_total = xp.concatenate((Y_ones, Y_zeros), axis=-1)
        costs = self.discriminator.train(X_total, Y_total, num_epochs, mini_batch_size, 2)
        return costs

    def save_weights(self, path, is_cupy=True):
        self.discriminator.save_weights(path+"/discriminator/", is_cupy)
        self.generator.save_weights(path+"/generator/", is_cupy)

    def load_weights(self, path):
        self.discriminator.load_weights(path+"/discriminator/")
        self.generator.load_weights(path+"/generator/")

    def train(self, X, num_epochs, gen_epochs=5, discrim_epochs=3, mini_batch_size=0):
        print_ind = max(num_epochs // 20, 1)
        m = X.shape[-1]
        total_discriminator_costs = []
        total_generator_costs = []
        for i in range(num_epochs):
            discriminator_costs = self.train_discriminator(X, discrim_epochs, mini_batch_size)
            generator_costs = self.train_generator(gen_epochs, m, mini_batch_size)
            if i % print_ind == 0:
                total_discriminator_costs.append(discriminator_costs[-1])
                total_generator_costs.append(generator_costs[-1])
                print(f"GAN: iteration: {i}, costs:\n\tdiscriminator: {discriminator_costs[-1]}\n\tgenerator: {generator_costs[-1]}")
                #if i % 5 == 0:
                    #out = xp.asnumpy(self.generate(1).T) 
                    #out = np.concatenate((out, out, out), axis=-1)
                    #plt.imsave(f"gen {i}.png", out[0].reshape(64,64,3), cmap="gray")
        return total_discriminator_costs, total_generator_costs

class DLVAEGANModel(DLGANModel):

    def __init__(self, name, generator, discriminator):
        super().__init__(name, generator, discriminator)

    def __str__(self):
        s = f"{self.name} VAE-GAN model description:"
        s += f"\tgenerator: {self.generator}"
        s += f"\tdiscriminator: {self.discriminator}"
        return s

    def generate(self, m):
        vae_layer = self.generator.bottleneck_layer
        n = self.generator.layers[vae_layer].input_shape[0]#_num_units
        out = xp.random.normal(0., 1., (n, m))
        out = self.generator.layers[vae_layer].forward_propagation(out, False)#activation_forward(out)
        for l in range(vae_layer+1, len(self.generator.layers)):
            out = self.generator.layers[l].forward_propagation(out, False)
        return out

    def compile(self, loss='wasserstein', gan_loss_weight=1.):
        super().compile(loss=loss)
        self.gan_loss_weight = gan_loss_weight

    def generator_minimax_loss_forward(self, Gz, x, m=0):
        vae_loss = xp.sum(self.generator.loss_forward(Gz, x))/m
        gan_loss = super().generator_loss_forward(Gz, m) * self.gan_loss_weight
        return vae_loss + gan_loss

    def generator_wasserstein_loss_forward(self, Gz, x, m=0):
        vae_loss = xp.sum(self.generator.loss_forward(Gz, x))/m
        gan_loss = super().generator_wasserstein_loss_forward(Gz, m) * self.gan_loss_weight 
        return vae_loss + gan_loss

    def generator_vae_loss_backward(self, Gz, x):
        return self.generator.loss_backward(Gz, x)

    def train_generator(self, X, Y, num_epochs, m, mini_batch_size=0):
        print_ind = max(num_epochs // 1, 1)
        L = len(self.generator.layers)
        if mini_batch_size == 0:
            mini_batch_size = m
        costs = []
        for i in range(num_epochs):
            Ji = 0
            mini_batches = self.generator.random_mini_batches(X, Y, mini_batch_size, i)
            for k in range(len(mini_batches)):
                #forward propagation
                Al = mini_batches[k][0]
                Yk = mini_batches[k][1]
                for l in range(1, L):
                    Al = self.generator.layers[l].forward_propagation(Al, True)
                #backward propagation
                D_Gz = self.discriminator.forward_propagation(Al)
                #print(xp.mean(D_Gz))
                dAl = self.generator_loss_backward(D_Gz)
                #print(dAl)
                for l in reversed(range(1, len(self.discriminator.layers))):
                    dAl = self.discriminator.layers[l].backward_propagation(dAl)
                #print(dAl)
                dAl += self.generator_vae_loss_backward(Al, Yk)
                for l in reversed(range(1, L)):
                    dAl = self.generator.layers[l].backward_propagation(dAl)
                    #update parameters
                    self.generator.layers[l].update_parameters(i+1)
                Ji += self.generator_loss_forward(Al, Yk, m)
            #record progress
            if i >= 0 and i % print_ind == 0:
                costs.append(Ji)
                print(self.report_generator_train_stats(i,Ji))
        return costs

    def train(self, X, num_epochs, gen_epochs=5, discrim_epochs=3, mini_batch_size=0, show_images=False, epoch_index=0):
        print_ind = 3#max(num_epochs // 250, 1)
        if show_images:
            plt.ion()
            fig, axs = plt.subplots(4, 5, figsize=(10, 8))
            axs = axs.flatten()
            plt.show()
        m = X.shape[-1]
        total_discriminator_costs = []
        total_generator_costs = []
        for i in range(epoch_index, num_epochs):
            discriminator_costs = self.train_discriminator(X, discrim_epochs, mini_batch_size)
            generator_costs = self.train_generator(X, X, gen_epochs, m, mini_batch_size)
            if i % print_ind == 0:
                total_discriminator_costs.append(discriminator_costs[-1])
                total_generator_costs.append(generator_costs[-1])
                print(f"GAN: iteration: {i}, costs:\n\tdiscriminator: {discriminator_costs[-1]}\n\tgenerator: {generator_costs[-1]}")
                if show_images:
                    plt.pause(0.1)
                    plt.cla()
                    out = xp.asnumpy(self.generate(20).T)
                    for j in range(20):
                        axs[j].imshow(out[j].reshape(64, 64, 3), cmap=matplotlib.cm.binary)
                        axs[j].axis('off')  # Turn off axis labels to improve visualization
                    if i % 3 == 0:
                        plt.savefig(f"gen {i//10}.png", bbox_inches="tight")
                        plt.pause(1.)
                    plt.show()
                if i % 50 == 0 and i > 0:
                    self.save_weights(f"model {i}", True)
                #if i % 5 == 0:
                    #out = xp.asnumpy(self.generate(1).T) 
                    #out = np.concatenate((out, out, out), axis=-1)
                    #plt.imsave(f"gen {i}.png", out[0].reshape(64,64,3), cmap="gray")
        return total_discriminator_costs, total_generator_costs