from model import get_data, normalize_data, reformat_data_as_time_series, train_model, split_train_test
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from timeit import default_timer as timer
from sklearn.preprocessing import MinMaxScaler



def plot_performance(params, history, model, training_data, testing_data):

    step = params["step"]

    print(f"Step is ", step)
    print(f"Epoch is ", params["epochs"])
    x_train, y_train = reformat_data_as_time_series(training_data, step)
    x_test, y_test = reformat_data_as_time_series(testing_data, step)

    print(f"Size of x_test ", len(x_test))
    print(f"Size of x_train ", len(x_train))

    # reshape data for model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    print(x_train.shape)
    # compute predicted outputs for train and test data
    predict_train = model.predict(x_train)
    predict_test = model.predict(x_test)

    # compute loss for train and test data
    train_loss = history.history["loss"][-1]
    test_loss = model.evaluate(x=x_test, y=y_test, verbose=0)

    # plot and save to png
    fig, axs = plt.subplots(ncols=2)
    ax = axs[0]
    ax.grid()
    ax.plot(np.arange(0, len(training_data)), training_data, color="tab:blue")
    ax.plot(np.arange(0, len(testing_data))+split_index, testing_data, color="tab:blue")
    ax.plot(np.arange(len(predict_train))+step, predict_train, color="tab:orange")
    ax.plot(np.arange(len(predict_test))+split_index+step, predict_test, color="tab:orange")
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.axvline(split_index, c="black", ls="--")
    ax = axs[1]
    ax.grid()
    ax.plot(history.history["loss"], label="train_loss")
    ax.plot(history.history["val_loss"], label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    title = "#x_test " + str(len(x_test)) + " step size = " + str(step) + " test loss " + str(round(test_loss,3)) + " train loss " + str(round(train_loss,3))
    ax.set_title(title)
    ax.legend(loc="center")
    filename = f"testloss-{int(test_loss)}_trainloss-{int(train_loss)}_" + "_".join([str(key) + "-" + str(value) for key, value in params.items()])
    fig.set_size_inches(300/25.4, 100/25.4)
    plt.savefig(f"train/{filename}.png", dpi=300)
    return train_loss, test_loss

TRAIN_TEST_SPLIT = 0.8


params = {
    "rnn_units": [50],
    "rnn_activation": ["tanh"], # ["relu", "tanh"],
    "step": [90],
    "dense_neurons": [20],
    "dense_activation": ["relu"], # ["relu", "tanh"],
    #"epochs": [50, 100, 150, 200, 250, 300, 350],
    "epochs": [1],
    "batch_size": [32],
    "optimizer": ["adam"],
    "learning_rate": [0.0005],
}

#params = {
#    "rnn_units": [50],
#    "rnn_activation": ["tanh"], # ["relu", "tanh"],
#    "step": [30, 60, 90],
#    "dense_neurons": [20],
#    "dense_activation": ["relu"], # ["relu", "tanh"],
#    #"epochs": [50, 100, 150, 200, 250, 300, 350],
#    "epochs": [25, 50],
#    "batch_size": [32],
#    "optimizer": ["adam"],
#    "learning_rate": [0.0005],
#}


# read the data
dataset = get_data()
dataset_scaled = normalize_data(dataset)

# create all combinations
all_names = params
combinations = it.product(*(params[name] for name in all_names))
combinations = list(combinations)
for i in range(len(combinations)):
    combinations[i] = {k:v for k, v in zip(all_names, combinations[i])}

print(f"there are {len(combinations)} combinations")

# Split into training and testing data
split_index = int(dataset.shape[0] * TRAIN_TEST_SPLIT)
training_data, testing_data = split_train_test(dataset_scaled, split_index)

# train
i = 0
start = timer()
models = []
for combination in combinations:
    i += 1
    print(f"####train start {i}/{len(combinations)}", combination)

    # train a network for each parameter combination
    history, model = train_model(combination, training_data)

    #save the model, you dont need to train it again if you want to use it further
    models.append([history, model])
    passed = timer()-start
    avg = passed / i
    total = avg * len(combinations)
    remain = total-passed

    print(f"####train finish, time passed = {passed:.1f} seconds, avg per combination = {avg:.1f} seconds, predicted = {(total/60):.3f} minutes, remain = {(remain/60):.3f} minutes")

## test and plot
#i = 0
#ratios = []
#for combination in combinations:
#    # plot the performance values
#    train_loss, test_loss = plot_performance(combination, models[i][0], models[i][1], training_data, testing_data)
#    print(f"train loss = ", train_loss )
#    print(f"test loss = ", test_loss )
#    combination["test loss"] = test_loss
#    combination["train loss"] = train_loss
#    combination["ratio"] = test_loss/train_loss
#    i += 1


import pandas as pd
df = pd.DataFrame.from_dict(combinations)
df = df.drop(['rnn_units', 'rnn_activation', 'dense_neurons', 'batch_size', 'learning_rate'], axis=1)

for dfIter in range(len(df)) :
    predictions = []
    step = df.loc[dfIter, "step"]
    first_batch = training_data[-step:]

    current_batch = first_batch.reshape((1, step, 1))

    for i in range (len(testing_data)):
        current_pred = models[dfIter][1].predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch_update = current_batch[:, 1:, :]
        print("mi")
        current_batch = np.append(current_batch_update, [[current_pred]], axis=2)
        print("sdsfdsfsfdfdsfdsfdsfdsfds")

    scaler = MinMaxScaler(feature_range=(0, 1))
    predictions_inverse_scaled = scaler.inverse_transform(predictions)
    test_inverse_scaled = scaler.inverse_transform(testing_data)
    plt.plot(predictions_inverse_scaled, label='predictions')
    plt.plot(test_inverse_scaled, label='test')
    plt.legend()
    title = "step=" + str(step) + " epochs=" + str(df.loc[dfIter, "epochs"])
    plt.title(title)
    plt.show()
