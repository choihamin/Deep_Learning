import numpy as np
from skimage.util.shape import view_as_windows


#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        #######
        ## If necessary, you can define additional class variables here
        #######

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        out_dim = input_size - filter_width + 1
        y = view_as_windows(x, (1, in_ch_size, filter_width, filter_height))
        y = y.reshape((batch_size, out_dim, out_dim, -1))
        w = self.W.reshape(num_filters, -1, 1)

        z = y @ w.T
        z = np.swapaxes(np.swapaxes(z, 3, 1), 2, 3)
        z += self.b

        return z

    #######
    # Q2. Complete this method
    #######
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
        dLdy = np.ones(y1.shape)
        batch_size, num_filters, out_height, out_width = dLdy.shape  # (16, 8, 28, 28)
        pad_height = filter_height - 1
        pad_width = filter_width - 1
        dLdy = np.pad(dLdy, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)), 'constant',
                      constant_values=0)
        y = view_as_windows(dLdy, (1, 1, filter_width, filter_height))  # (16, 8, 32, 32, 1, 1, 5, 5)
        y = y.reshape((batch_size, num_filters, input_size, input_size, -1))  # (16, 8, 32, 32, 25)
        filt = np.flip(np.flip(W, axis=3), axis=2)  # (8, 3, 25,1)
        filt = filt.reshape((num_filters, in_ch_size, -1, 1))

        dLdx = np.zeros(x.shape)
        for i in range(batch_size):
            for j in range(num_filters):
                for k in range(in_ch_size):
                    e = np.squeeze(y[i][j] @ filt[j][k], axis=2)
                    dLdx[i][k] += e
        ##########################################

        print(dLdx.shape)
        print(dLdW.shape)
        print(dLdb.shape)

        return dLdx, dLdW, dLdb

    #######
    ## If necessary, you can define additional class methods here
    #######


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q3. Complete this method
    #######
    def forward(self, x):
        batch_size, in_ch_size, in_height, in_width = x.shape
        out = [None for _ in range(batch_size)]
        for i, batch in enumerate(x):
            y = view_as_windows(batch, (in_ch_size, self.pool_size, self.pool_size), step=self.stride)
            y = y.reshape((-1, in_ch_size, self.pool_size ** 2))
            out[i] = np.array([[e.max() for e in d] for d in y]).T.reshape(in_ch_size, in_height // self.pool_size,
                                                                           in_width // self.pool_size)  # 이부분 한번 더 보자

        return np.array(out)

    #######
    # Q4. Complete this method
    #######
    def backprop(self, x, dLdy):
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

        return np.array(dLdx)

    #######
    ## If necessary, you can define additional class methods here
    #######


# testing the implementation

# data sizes
batch_size = 16
input_size = 32
filter_width = 5
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')