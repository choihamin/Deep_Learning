import numpy as np
from skimage.util.shape import view_as_windows
from skimage.measure import block_reduce

##########
#   convolutional layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):

        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        out_ch_size, in_ch_size, Wx_size, Wy_size = self.W.shape
        batch_size, _, in_height, in_width = x.shape
        input_size = self.input_size
        out_dim_x = input_size - Wx_size + 1
        out_dim_y = input_size - Wy_size + 1
        y = view_as_windows(x, (1, in_ch_size, Wx_size, Wy_size))
        y = y.reshape((batch_size, out_dim_x, out_dim_y, -1))
        w = self.W.reshape(out_ch_size, -1, 1)

        z = y @ w.T
        z = np.swapaxes(np.swapaxes(z, 3, 1), 2, 3)
        z += self.b

        return z

    def backprop(self, x, dLdy):

        num_filters, in_ch_size, filter_width, filter_height = self.W.shape  # W : 8 x 3 x 5 x 5
        batch_size, _, in_height, in_width = x.shape  # x : 16 x 3 x 32 x 32
        W = self.W
        _, _, out_height, out_width = dLdy.shape

        ################# dLdW ###################
        y = view_as_windows(x, (1, 1, out_height, out_width))  # (16, 3, 5, 5, 1, 1, 28, 28)
        y = y.reshape((batch_size * in_ch_size, filter_height, filter_width, -1))  # (48, 5, 5, 784)
        filt = dLdy.reshape(batch_size * num_filters, 1, out_height, out_width)  # (128, 1, 28, 28)
        filt = filt.reshape((batch_size * num_filters, -1, 1))  # (128, 784, 1)
        z = y @ filt.T  # (48, 5, 5, 128)
        z = np.swapaxes(np.swapaxes(z, 3, 1), 2, 3)  # (48, 128, 5, 5)
        z = z.reshape((batch_size, in_ch_size, batch_size * num_filters, filter_width, filter_height)).sum(
            axis=0)  # (16, 128, 5, 5)
        z = z.reshape((in_ch_size, batch_size, num_filters, filter_width, filter_height)).sum(axis=1)
        dLdW = np.swapaxes(z, 0, 1) / batch_size
        ##########################################

        ################# dLdb ###################
        dLdb = dLdy.sum(axis=0).sum(axis=1).sum(axis=1).reshape(1, -1, 1, 1)
        ##########################################

        ################# dLdx ###################
        dLdy = np.ones(dLdy.shape)
        batch_size, num_filters, out_height, out_width = dLdy.shape  # (16, 8, 28, 28)
        pad_height = filter_height - 1
        pad_width = filter_width - 1
        dLdy = np.pad(dLdy, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)), 'constant',
                      constant_values=0)
        y = view_as_windows(dLdy, (1, 1, filter_width, filter_height))  # (16, 8, 32, 32, 1, 1, 5, 5)
        y = y.reshape((batch_size, num_filters, in_width, in_height, -1))  # (16, 8, 32, 32, 25)
        filt = np.flip(np.flip(W, axis=3), axis=2)  # (8, 3, 5, 5)
        filt = filt.reshape((num_filters, in_ch_size, -1, 1))  # (8, 3, 25, 1)

        y = np.swapaxes(y, 1, 3)
        y = np.expand_dims(y, axis=4)
        filt = np.swapaxes(filt, 0, 1)

        dLdx = np.zeros((in_ch_size, batch_size, in_height, in_width))
        for k in range(in_ch_size):
            g = y @ filt[k]
            g = np.squeeze(np.squeeze(g.sum(axis=3), axis=-1), axis=-1)
            dLdx[k] = g
        dLdx = np.swapaxes(np.swapaxes(dLdx, 2, 3), 0, 1)

        """for i in range(batch_size):
            for j in range(num_filters):
                for k in range(in_ch_size):
                    e = np.squeeze(y[i][j] @ filt[j][k], axis=2)
                    dLdx[i][k] += e"""
        ##########################################
        return dLdx, dLdW, dLdb

##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        batch_size, in_ch_size, in_height, in_width = x.shape
        out = [None for _ in range(batch_size)]
        for i, batch in enumerate(x):
            y = view_as_windows(batch, (in_ch_size, self.pool_size, self.pool_size), step=self.stride)
            y = y.reshape((-1, in_ch_size, self.pool_size ** 2))
            out[i] = np.array([[e.max() for e in d] for d in y]).T.reshape(in_ch_size, in_height // self.pool_size,
                                                                           in_width // self.pool_size)  # 이부분 한번 더 보자

        return np.array(out)

    def backprop(self, x, dLdy):
        out = block_reduce(x, (1, 1, 2, 2), np.max)
        mask = np.equal(x, out.repeat(2, axis=2).repeat(2, axis=3)).astype(int)
        dLdx = np.multiply(mask, dLdy.repeat(2, axis=2).repeat(2, axis=3))
        return dLdx

    """def backprop(self, x, dLdy):

        dLdy = dLdy.reshape((-1, 28, 13, 13))
        dLdy = np.swapaxes(dLdy, 2, 3)
        batch_size, in_ch_size, out_width, out_height = dLdy.shape
        _, _, in_height, in_width = x.shape
        dLdx = [None for _ in range(batch_size)]
        for j, batch in enumerate(x):
            out = [None for _ in range(in_ch_size)]
            upstream = dLdy[j]
            for k, chn in enumerate(batch):
                upstream_mat = upstream[k]
                y = view_as_windows(chn, (self.pool_size, self.pool_size), step=self.stride)
                y = y.reshape(-1, self.pool_size ** 2)
                y = np.array([[1 if a == e.max() else 0 for a in e] for e in y]).reshape(
                    (-1, self.pool_size, self.pool_size))
                c = upstream_mat.flatten()
                y = np.array([y[i] * c[i] for i in range(len(y))])
                lst = []
                for i, e in enumerate(y):
                    if i % (in_width // self.pool_size) == 0:
                        tmp = e
                    elif i % (in_width // self.pool_size) == (in_width // self.pool_size) - 1:
                        tmp = np.hstack((tmp, e))
                        if len(lst) != 0:
                            lst = np.vstack((lst, tmp))
                        else:
                            lst = tmp
                        tmp = None
                    else:
                        tmp = np.hstack((tmp, e))
                out[k] = lst
            dLdx[j] = out
        

        return np.array(dLdx)"""



##########
#   fully connected layer
##########
# fully connected linear layer.
# parameters: weight matrix matrix W and bias b
# forward computation of y=Wx+b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size,)

class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b = 0.01+np.zeros((output_size))

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        out = x @ self.W.T + self.b.T
        return out

    def backprop(self,x,dLdy):
        shape = x.shape
        x = x.reshape((x.shape[0], -1))

        dLdx = np.zeros(x.shape)
        dLdW = np.zeros_like(self.W)
        dLdb = np.zeros_like(self.b)

        for n in range(x.shape[0]):
            dLdW += np.outer(dLdy[n], x[n])
            dLdb += dLdy[n]
            dLdx[n] = dLdy[n] @ self.W
        dLdx = dLdx.reshape(shape)
        return dLdx, dLdW, dLdb

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

class nn_activation_layer:

    # performs ReLU activation
    def __init__(self):
        pass

    def forward(self, x):
        x[x<=0] = 0
        return x


    def backprop(self, x, dLdy):
        x[x<=0] = 0
        x[x>0] = 1
        dLdx = np.multiply(x, dLdy)

        return dLdx


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self, x):
        exp = np.exp(x)
        return np.array(list(map(lambda x: x / np.sum(x), exp)))

    def backprop(self, x, dLdy):
        batch_size, _ = x.shape
        dLdx = np.zeros(x.shape)
        for n in range(batch_size):
            y = np.exp(x[n]) / np.sum(np.exp(x[n]))
            y = y.reshape(-1, 1)
            dydx = np.diagflat(y) - y @ y.T
            dLdx[n] = dLdy[n] @ dydx
        return dLdx

##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:

    def __init__(self):
        pass

    def forward(self, x, y):
        prob = x[:, 0]
        prob = prob.reshape(y.shape[0], 1)
        L = -(np.multiply((1-y),np.log(prob)) + np.multiply(y,np.log(1-prob)))
        return np.mean(L)

    def backprop(self, x, y):
        batch_size = y.shape[0]
        dLdy = 1 / batch_size
        dydx = np.zeros_like(x)
        for n in range(batch_size):
            dydx[n][y[n]] = -1 / x[n][y[n]]
        dLdx = np.multiply(dLdy, dydx)
        return dLdx
