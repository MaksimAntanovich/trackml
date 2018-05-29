import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import pickle

from data_path import DATA_PATH

data_ctx = mx.gpu()
model_ctx = mx.gpu()

X = pickle.load(open(DATA_PATH + '/train_X.pkl', 'rb'))
print(X.shape)
y = pickle.load(open(DATA_PATH + '/train_Y.pkl', 'rb'))
print(y.shape)

num_inputs = X.shape[1]
num_outputs = y.shape[1]

batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X[2500:], y[2500:]),
                                      batch_size=batch_size, shuffle=True)

test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X[:2500], y[:2500]),
                                      batch_size=batch_size, shuffle=True)

num_hidden = 16
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_inputs, activation="relu", dtype='float64'))
    net.add(gluon.nn.Dense(num_hidden, activation="relu", dtype='float64'))
    net.add(gluon.nn.Dense(num_outputs, dtype='float64'))

net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

softmax_cross_entropy = gluon.loss.L1Loss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.RMSE()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        output = net(data)
        acc.update(preds=output, labels=label)
    return acc.get()[1]

epochs = 10
smoothing_constant = .01

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_loss += nd.sum(loss).asscalar()


    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, cumulative_loss, train_accuracy, test_accuracy))
