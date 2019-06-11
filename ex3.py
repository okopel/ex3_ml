#################################
# Ori Kopel, Shlomo Rabinovich  #
# 205533151, 308432517          #
#################################
import matplotlib.pyplot as plt
import numpy as np


class NeuralNet(object):
    """
    create 2 layers neural network, which has an hidden layer.
    classificate to 10 classes of clothes.
    using ReLU function.
    """

    def __init__(self, data_size, numOfClasses, numOfLayers):
        self.w1 = np.random.randn(data_size, numOfLayers) * 0.0001
        self.w2 = np.random.randn(numOfLayers, numOfClasses) * 0.0001
        self.bias1 = np.zeros((1, numOfLayers))
        self.bias2 = np.zeros((1, numOfClasses))

    def loss(self, train_x, label_y, reg):
        """
        calc loss & gradient for our net.

        params:
        X: training set.
        y: training labels.
        if y=none, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return prediction of X.

        else,
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        w1 = self.w1
        bias1 = self.bias1
        w2 = self.w2
        bias2 = self.bias2
        x_size, y_size = train_x.shape

        # Compute the forward pass
        layer = norm_relu(np.dot(train_x, w1) + bias1)
        scores = np.dot(layer, w2) + bias2

        # If the targets are not given then jump out, we're done
        if label_y is None:
            return scores

        # Compute the loss
        exp_scores = np.exp(scores - (np.max(scores, axis=1, keepdims=True)))
        pr = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        c_log_pr = -np.log(pr[range(x_size), label_y])
        reg_loss_func = ((reg * np.sum(w1 * w1)) / 2) + ((reg * np.sum(w2 * w2)) / 2)
        data_loss_func = np.sum(c_log_pr) / x_size
        loss = data_loss_func + reg_loss_func

        # Backward pass: compute gradients
        pr[range(x_size), label_y] -= 1
        pr /= x_size
        d_w2 = np.dot(layer.T, pr)
        db2 = np.sum(pr, axis=0, keepdims=True)
        dh1 = np.dot(pr, w2.T)
        dh1[layer <= 0] = 0
        d_w1 = np.dot(train_x.T, dh1)
        db1 = np.sum(dh1, axis=0, keepdims=True)
        d_w2 += reg * w2
        d_w1 += reg * w1
        return loss, {'w1': d_w1, 'bias1': db1, 'w2': d_w2, 'bias2': db2}

    def train(self, train_x, train_label, x_val, y_val, param):

        """
        params:
        X: training set.
        y: training set labels
        X_val: validation data.
        y_val: validation labels.
        lr_change: used to updatelearning rate after epoch.
        reg: regularization strength.
        verbose: debug mode.
        """
        lr = param['learning_rate']
        lr_change = param['learning_rate_decay']
        reg = param['reg']
        mu = param['mu']
        epoch = param['epochs']
        mu_increase = param['mu_increase']
        batch_size = param['batch_s']
        verbose = param['verbose']
        num_train = train_x.shape[0]
        iterations_per_epoch = max(int(num_train / batch_size), 1)

        # Use SGD to optimize the parameters in self.model
        w22, b22, w11, b11 = 0.0, 0.0, 0.0, 0.0
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for i in range(1, epoch * iterations_per_epoch + 1):
            # shuffle and make batch
            sample_index = np.random.choice(num_train, batch_size, replace=True)
            x_batch = train_x[sample_index, :]
            y_batch = train_label[sample_index]

            # calc gradients & loss
            loss_func, gradients = self.loss(x_batch, label_y=y_batch, reg=reg)
            loss_history.append(loss_func)
            w1 = gradients['w1']
            w2 = gradients['w2']
            b1 = gradients['bias1']
            b2 = gradients['bias2']

            # update by gradients
            w22 = (mu * w22) - (w2 * lr)
            b22 = (mu * b22) - (b2 * lr)
            w11 = (mu * w11) - (w1 * lr)
            b11 = (mu * b11) - (b1 * lr)
            self.w1 += w11
            self.bias1 += b11
            self.w2 += w22
            self.bias2 += b22

            # todo del verbose
            if verbose and i % iterations_per_epoch == 0:
                # Every epoch, check train and val accuracy and decay learning rate.
                epoch = i / iterations_per_epoch
                train_acc = (self.predict(x_batch) == y_batch).mean()
                v_acc = (self.predict(x_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(v_acc)
                print("epoch %d / %d: loss %f, train_acc: %f, val_acc: %f" %
                      (epoch, epoch, loss_func, train_acc, v_acc))

                # Decay learning rate
                lr *= lr_change
                mu *= mu_increase

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, vec):
        # calc in first layer
        h1 = norm_relu(np.dot(vec, self.w1) + self.bias1)
        # calc in second layer
        scores = np.dot(h1, self.w2) + self.bias2
        # choose the class with highest score.
        y_pred = np.argmax(scores, axis=1)
        return y_pred


def norm_relu(x):
    # ReLU function
    return np.maximum(0, x)


def showPlt(states):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.subplots_adjust(wspace=0, hspace=0.3)

    # Plot the loss function and train / validation accuracies

    plt.subplot(2, 1, 1)
    plt.plot(states['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(states['train_acc_history'], label='train')
    plt.plot(states['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load the data
    train_X = np.loadtxt("train_x", dtype=np.uint8)
    train_y = np.loadtxt("train_y", dtype=np.uint8)
    # y_test
    test_x = np.loadtxt("test_x", dtype=np.uint8)
    train_size = train_X.shape[0]
    test_size = test_x.shape[0]

    # create validation set
    validation_size = 1000
    train_size -= validation_size

    index = list(range(train_size, validation_size + train_size))
    validation_x = train_X[index]
    validation_y = train_y[index]

    index = list(range(train_size))
    train_X = train_X[index]
    train_y = train_y[index]

    index = list(range(test_size))
    test_x = test_x[index]
    # y_test = y_test[mask]
    train_X = train_X.reshape(train_size, -1)
    validation_x = validation_x.reshape(validation_size, -1)
    test_x = test_x.reshape(test_size, -1)

    net = NeuralNet(train_X.shape[1], numOfClasses=10, numOfLayers=10)

    params = {'epochs': 10, 'batch_s': 1024, 'learning_rate': 0.00075,
              'learning_rate_decay': 0.95, 'reg': 1.0, 'mu': 0.9, 'mu_increase': 1.0, 'verbose': True}
    # Train the network
    result = net.train(train_X, train_y, validation_x, validation_y, params)

    # Predict on the validation set
    val_acc = (validation_y == net.predict(validation_x)).mean()
    showPlt(result)
