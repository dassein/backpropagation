import numpy as np
import matplotlib.pyplot as plt


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - x * x


def train(X, Y, lr=2., epoch=5000, err_break=1e-3,
          wo=np.ones((13, 1)), w1=np.ones((5, 1)), w2=np.ones((5, 1))):
    list_err = []
    for num_epoch in range(epoch):
        loss = 0
        for (x_tmp, y) in list(zip(X, Y)):
            len_input = len(w1) - 1
            len_hidden = len(x_tmp) - (len_input - 1)
            # Forward
            # v = [v1, v2, 1]; x = [..., 1]
            # net1: v1 = sigmoid(z1); z1 = x @ w1
            # net2: v2 = sigmoid(z2); z2 = x @ w2
            # out : o  = sigmoid(z ); z  = v @ wo
            # loss: sum( (y - o) * (y - o)/ 2. )
            x = []
            for start in range(len_hidden):
                end = start + len_input
                x.append(x_tmp[start:end])
            x  = np.hstack((np.asarray(x), np.ones((len(x), 1))))
            v1 = tanh(x @ w1)  # v1
            v2 = tanh(x @ w2)  # v2
            v = np.vstack((v1, v2))
            v = np.concatenate((v, [[1]]))
            o = tanh(v.transpose() @ wo)
            loss = loss + ((y - o) * (y - o))[0][0] / 2.
            # Backward
            # For o, v
            # d loss / d o = (o - y)
            # d loss / d v = (d z / d v)(d loss / d z) = wo * (d loss / d z)
            # For z, z1, z2
            # d o  / d z  = sigmoid_prime(o )
            # d v1 / d z1 = sigmoid_prime(v1)
            # d v2 / d z2 = sigmoid_prime(v2)
            # d loss / d z  = (d o  / d z )(d loss / d o )   = (d o  / d z ) * (o - y)
            #               = sigmoid_prime(o ) * (o - y)
            # d loss / d z1 = (d v1 / d z1)(d loss / d v)[1] = (d v1 / d z1) * (d loss / d v)[1]
            #               = sigmoid_prime(v1) * { wo * (d loss / d z) }[1]
            #               = sigmoid_prime(v1) * wo[1] * (d loss / d z)
            # d loss / d z2 = (d v2 / d z2)(d loss / d v)[2] = (d v2 / d z2) * (d loss / d v)[2]
            #               = sigmoid_prime(v2) * { wo * (d loss / d z) }[2]
            #               = sigmoid_prime(v2) * wo[2] * (d loss / d z)
            delta_zo = tanh_prime(o) * (o - y)  # d loss / d z
            delta_z1 = tanh_prime(v1) * (wo @ delta_zo)[0:len_hidden]  # d loss / d z1
            delta_z2 = tanh_prime(v2) * (wo @ delta_zo)[len_hidden:-1]  # d loss / d z2
            # For wo, w1, w2
            # d loss / d wo =  (d z  / d wo)(d loss / d z ) = v  * (d loss / d z )
            # d loss / d w1 =  (d z1 / d w1)(d loss / d z1) = x' * (d loss / d z1)
            # d loss / d w2 =  (d z1 / d w2)(d loss / d z2) = x' * (d loss / d z2)
            delta_wo = v @ delta_zo  # d loss / d wo
            delta_w1 = x.transpose() @ delta_z1  # d loss / d w1
            delta_w2 = x.transpose() @ delta_z2  # d loss / d w2
            # update: w := w - lr * (d loss / d w)
            wo = wo - lr * delta_wo
            w1 = w1 - lr * delta_w1
            w2 = w2 - lr * delta_w2
        list_err.append(loss)
        if loss < err_break:
            print('Run ' + str(num_epoch) + ' epochs')
            break
    return wo, w1, w2, list_err


if __name__ == "__main__":
    X = np.array([
        [0, 0, 0.8, 0.4, 0.4, 0.1, 0, 0, 0],
        [0, 0.3, 0.3, 0.8, 0.3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.3, 0.3, 0.8, 0.3, 0],
        [0, 0, 0, 0, 0, 0.8, 0.4, 0.4, 0.1],
        [0.8, 0.4, 0.4, 0.1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.3, 0.3, 0.8, 0.3],
    ])
    Y = np.asarray([[-1, 1, 1, -1, -1, 1]]).transpose()
    wo_init = np.asarray([[1.20973877, -1.07518386, 0.80691921, -0.29078347, -0.22094764,
                           -0.16915604, 1.10083444, 0.08251052, -0.00437558, -1.72255825,
                           1.05755642, -2.51791281, -1.91064012]]).transpose()
    w1_init = np.asarray([[1.73673761, 1.89791391, -2.10677342, -0.14891209, 0.58306155]]).transpose()
    w2_init = np.asarray([[-2.25923303, 0.13723954, -0.70121322, -0.62078008, -0.47961976]]).transpose()
    wo, w1, w2, list_err = train(X, Y, lr=0.2, epoch=1000, wo=wo_init, w1=w1_init, w2=w2_init)
    print('hidden layer 1, neuron 1 weights\n', w1)
    print('hidden layer 1, neuron 2 weights\n', w2)
    print('hidden layer 2, neuron 1 weights\n', wo)

    plt.plot(list_err)
    plt.ylabel('error')
    plt.xlabel('epochs')
    plt.show()
    for ind, (x_tmp, y) in enumerate(list(zip(X, Y))):
        len_input = len(w1) - 1
        len_hidden = len(x_tmp) - (len_input - 1)
        x = []
        for start in range(len_hidden):
            end = start + len_input
            x.append(x_tmp[start:end])
        x = np.hstack((np.asarray(x), np.ones((len(x), 1))))
        v1 = tanh(x @ w1)  # v1
        v2 = tanh(x @ w2)  # v2
        v = np.vstack((v1, v2))
        v = np.concatenate((v, [[1]]))
        o = tanh(sum(v * wo))
        print(str(ind) + ": produced: " + str(o) + " wanted " + str(y))