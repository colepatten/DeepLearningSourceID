import os
import re
import csv
import tensorflow as tf
from keras import layers
from keras import activations
from keras import models
from keras import callbacks
from keras import regularizers
from keras.optimizers import AdamW
from keras.initializers import HeNormal
from keras.metrics import AUC

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# set hyperparameters
period = 20
num_epochs = 20000
smoothing_window_size = 10 # for smoothing the accuracy and loss plots
BATCH_SIZE = 989 
METRIC = "angular" # using angular distance


# function to smooth plots
def smooth_curve(data, smoothing_window=10):
    """Applies a simple moving average to smooth the curve."""
    if len(data) < smoothing_window:
        return data  # Not enough data to smooth
    return np.convolve(data, np.ones(smoothing_window) / smoothing_window, mode="same")

# plot AUC over time
def plot_auc_metrics(history, save_path, period=1, smoothing_window=10):
    # extract data from history
    loss_epochs = np.array(range(len(history.history.get("loss", []))))
    training_loss = history.history.get("loss", [])
    
    # smooth training loss
    smoothed_loss = smooth_curve(training_loss, smoothing_window_size)

    # extract AUC metrics
    acc_epochs = period * np.array(range(len(history.history.get("nbide auc", []))))
    nbide_auc = history.history.get("nbide auc", [])
    smoothed_nbide_auc = smooth_curve(nbide_auc, smoothing_window_size) if nbide_auc else []
    
    pop_auc = history.history.get("pop auc", [])
    smoothed_pop_auc = smooth_curve(pop_auc, smoothing_window_size) if pop_auc else []

    # create a figure with 2 rows and 2 columns for the plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.4)  # Adjust spacing between plots

    # plot training loss and smoothed loss
    axs[0, 0].plot(loss_epochs, training_loss, label="Training Loss", marker="o", color="blue")
    axs[0, 0].set_title("Training Loss Over Epochs")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_ylim(0, 10)
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(loss_epochs, smoothed_loss, label="Smoothed Loss", marker="o", color="blue")
    axs[0, 1].set_title("Smoothed Training Loss Over Epochs")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].set_ylim(0, 10)
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # plot NBIDE AUC and its smoothed version
    axs[1, 0].plot(acc_epochs, nbide_auc, label="NBIDE AUC", marker="o", color="blue")
    # Annotate the maximum NBIDE AUC
    max_nbide_index = np.argmax(nbide_auc)
    max_nbide_value = nbide_auc[max_nbide_index]
    axs[1, 0].annotate(f"Max: {max_nbide_value:.3f}",
                       (acc_epochs[max_nbide_index], max_nbide_value),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha="center",
                       fontsize=10,
                       color="red")
    axs[1, 0].set_title("NBIDE AUC Over Epochs")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("AUC")
    axs[1, 0].set_ylim(0.5, 1)
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    axs[1, 1].plot(acc_epochs, smoothed_nbide_auc, label="Smoothed NBIDE AUC", marker="o", color="blue")
    # annotate the maximum NBIDE AUC
    max_nbide_index = np.argmax(smoothed_nbide_auc)
    max_nbide_value = smoothed_nbide_auc[max_nbide_index]
    axs[1, 1].annotate(f"Max: {max_nbide_value:.3f}",
                       (acc_epochs[max_nbide_index], max_nbide_value),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha="center",
                       fontsize=10,
                       color="red")
    axs[1, 1].set_title("NBIDE AUC Over Epochs")
    axs[1, 1].set_xlabel("Epochs")
    axs[1, 1].set_ylabel("AUC")
    axs[1, 1].set_ylim(0.5, 1)
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # save the plot
    plt.savefig(save_path)
    plt.close()  # Close the plot to free resources
    print(f"Plots saved to {save_path}")



# create a directory to save a new model to 
def get_next_model_directory(base_path):
    # List all directories in the base path that match the 'width_16' pattern
    existing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Use regular expressions to find directories matching the 'width_16' pattern
    model_numbers = []
    for d in existing_dirs:
        match = re.match(r"^width_16_(\d+)$", d)
        if match:
            # Extract the number from the directory name and add to the list
            num = int(match.group(1))
            model_numbers.append(num)
    
    # Get the next available number starting from 0
    next_model_num = max(model_numbers, default=-1) + 1  # Start from -1, so the first is 0
    new_dir = os.path.join(base_path, f"width_16_{next_model_num}")
    os.makedirs(new_dir, exist_ok=True)
    return new_dir
    
    
    

#import data
e3_skewed = np.load('/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/E3/processed_E3/e3_cylinders.npy')

path = "/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/E3/info.csv"
e3_info = pd.read_csv(path, encoding="latin_1")
e3_labels = np.asarray(e3_info['Firearm'])

e3_data = np.zeros_like(e3_skewed)
for i in range (e3_data.shape[0]):
    red = e3_skewed[i]
    zone = ~(red == 0.)
    mean = np.mean(red[zone])
    std = np.std(red[zone])
    e3_data[i] = np.where(zone, (red-mean)/std, red)
    
nbide_skewed = np.load('/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/images/processed_cylinders/processed_cylinders.npy')
path = "/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/NBIDE/info.csv"
nbide_info = pd.read_csv(path, encoding="latin_1")
nbide_labels = np.asarray(nbide_info['Firearm'])

nbide_data = np.zeros_like(nbide_skewed)
for i in range (nbide_data.shape[0]):
    red = nbide_skewed[i]
    zone = ~(red == 0.)
    mean = np.mean(red[zone])
    std = np.std(red[zone])
    nbide_data[i] = np.where(zone, (red-mean)/std, red)




# functions to evaluate distances between embedded images

def dists_and_mask(y_true, y_pred):
    tf.debugging.assert_all_finite(y_pred, "pred bad")
    y_true = tf.squeeze(y_true)
    same_identity_mask = tf.equal(tf.expand_dims(y_true, axis=1),
                                  tf.expand_dims(y_true, axis=0))
    negative_mask = tf.logical_not(same_identity_mask)
    positive_mask = tf.math.logical_xor(same_identity_mask,
                                   tf.eye(tf.shape(y_true)[0], dtype=tf.bool))

    dists = loss_metric(y_pred)
    tf.debugging.assert_all_finite(dists, "dists bad")

    return dists, positive_mask, negative_mask

def get_scores(data, labels):
    embedded_data = model.predict(data)
    test_dists, test_positive_mask, test_negative_mask = dists_and_mask(labels, embedded_data)
    positive_scores = tf.boolean_mask(test_dists, test_positive_mask)
    negative_scores = tf.boolean_mask(test_dists, test_negative_mask)
    return positive_scores, negative_scores
    
    

# calculate AUC for some embeddings and their true class labels

auc_metric = AUC(num_thresholds=10297)    
def base_auc_accuracy(embeddings, labels):
    labels = tf.squeeze(labels)

    same_barrel_mask = tf.equal(tf.expand_dims(labels, axis=1),
                                  tf.expand_dims(labels, axis=0))
    negative_mask = tf.logical_not(same_barrel_mask)
    positive_mask = tf.math.logical_xor(same_barrel_mask,
                                   tf.eye(tf.shape(labels)[0], dtype=tf.bool))

    dists = loss_metric(embeddings)

    positive_scores = tf.boolean_mask(dists, positive_mask)
    negative_scores = tf.boolean_mask(dists, negative_mask)
    
    scores = 1 - tf.concat([positive_scores, negative_scores], axis=0)
    truths = tf.concat([tf.ones_like(positive_scores), tf.zeros_like(negative_scores)], axis=0)
    
  
    auc_metric.reset_state()
    auc_metric.update_state(truths, scores)
    roc_auc = auc_metric.result().numpy()

    return roc_auc
    
    
# class for neural network to evaluate training and validation AUC after set amount of epochs    
class Metrics(callbacks.Callback):
    def __init__(self, embedding_model, x1, y1, logs={}):
        self.embedding_model = embedding_model
        self.x1 = x1
        self.y1 = y1
        self.nbide_auc = 0
        self.accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1) % period == 0:
            emb = self.embedding_model.predict(self.x1, verbose=False)
            self.nbide_auc = base_auc_accuracy(emb, self.y1)
            logs["nbide auc"] = self.nbide_auc


# defines a metric to calculate distances between embedded images
def loss_metric(embeddings):
    if METRIC == "L2":
        difference = tf.expand_dims(embeddings, axis=0) - tf.expand_dims(embeddings, axis=1)
        difference = tf.sqrt(tf.reduce_sum(tf.square(difference), axis=-1))
        return difference

    elif METRIC == "angular":
        embeddings = tf.math.l2_normalize(embeddings, axis=-1)
        cos_sim = tf.matmul(embeddings, embeddings, transpose_b=True)
        distance = 2*tf.acos(tf.clip_by_value(cos_sim, 0+1e-7, 1-1e-7)) / tf.constant(np.pi)
        return distance



# supervised contrastive loss function 
def base_supcon_loss_func(y_true, y_pred):
    tf.debugging.assert_all_finite(y_pred, "pred bad")
    y_true = tf.squeeze(y_true)
    same_identity_mask = tf.equal(tf.expand_dims(y_true, axis=1),
                                  tf.expand_dims(y_true, axis=0))
    negative_mask = tf.logical_not(same_identity_mask)
    positive_mask = tf.math.logical_xor(same_identity_mask,
                                   tf.eye(tf.shape(y_true)[0], dtype=tf.bool))

    embeddings = tf.math.l2_normalize(y_pred, axis=-1)
    cos_sim = tf.matmul(embeddings, embeddings, transpose_b=True)*10
    tf.debugging.assert_all_finite(cos_sim, "dists bad")

    negative_sims = tf.boolean_mask(cos_sim, negative_mask)
    positive_sims = tf.boolean_mask(cos_sim, positive_mask)

    negative_loss = tf.math.log( tf.reduce_sum( tf.math.exp(negative_sims)))
    positive_loss = tf.reduce_mean(positive_sims)
    loss = negative_loss - positive_loss
    return loss
    
    

# create model
class CyclicPadding2D(layers.Layer):
    def __init__(self, padding=1, **kwargs):
        super(CyclicPadding2D, self).__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        # Get the input shape
        input_shape = tf.shape(inputs)

        # Extract the cyclic padding segments
        left_pad = inputs[:, -self.padding:, :]
        right_pad = inputs[:, :self.padding, :]
        # Concatenate the padding segments with the input
        padded_inputs = tf.concat([left_pad, inputs, right_pad], axis=1)

        up_down_pad = tf.zeros_like(padded_inputs[:, :, :self.padding])
        padded_inputs = tf.concat([up_down_pad, padded_inputs, up_down_pad], axis=2)

        return padded_inputs

    def get_config(self):
        config = super(CyclicPadding2D, self).get_config()
        config.update({"padding": self.padding})
        return config
        

def build_flagship_test():
    initializer = HeNormal(seed=4815162342)       
    input = layers.Input(shape=(377,60, 1))

    y = CyclicPadding2D(padding=3)(input)
    y = layers.Conv2D(16, 7, activation='relu', padding="valid", kernel_initializer=initializer)(y)
    x = layers.MaxPool2D((2,1))(y)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(.5, seed=4)(x)
    x = layers.Conv2D(16, 1, activation='relu', padding="valid", kernel_initializer=initializer)(x)
    r = layers.MaxPool2D((4,2))(x)
    y = CyclicPadding2D(padding=3)(x)
    y = layers.Conv2D(16, 7, activation='relu', padding="valid", kernel_initializer=initializer)(y)
    y = layers.MaxPool2D((2,1))(y)
    y = CyclicPadding2D(padding=1)(y)
    y = layers.Conv2D(16, 3, activation='relu', padding="valid", kernel_initializer=initializer)(y)
    y = layers.MaxPool2D((2,2))(y)
    x = y + r

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(.5, seed=8)(x)
    x = layers.Conv2D(32, 1, activation='relu', padding="valid", kernel_initializer=initializer)(x)
    r = layers.MaxPool2D((4,2))(x)
    y = CyclicPadding2D(padding=2)(x)
    y = layers.Conv2D(16, 5, activation='relu', padding="valid", kernel_initializer=initializer)(y)
    y = layers.MaxPool2D((2,1))(y)
    y = CyclicPadding2D(padding=1)(y)
    y = layers.Conv2D(32, 3, activation='relu', padding="valid", kernel_initializer=initializer)(y)
    y = layers.MaxPool2D((2,2))(y)
    x = y + r

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(.5, seed=15)(x)
    x = layers.Conv2D(64, 1, activation='relu', padding="valid", kernel_initializer=initializer)(x)
    r = layers.MaxPool2D((4,2))(x)
    y = CyclicPadding2D(padding=2)(x)
    y = layers.Conv2D(16, 5, activation='relu', padding="valid", kernel_initializer=initializer)(y)
    y = layers.MaxPool2D((2,1))(y)
    y = CyclicPadding2D(padding=1)(y)
    y = layers.Conv2D(64, 3, activation='relu', padding="valid", kernel_initializer=initializer)(y)
    y = layers.MaxPool2D((2,2))(y)
    x = y + r

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(.5)(x)
    x = layers.GlobalMaxPool2D()(x) #128
    x = layers.Dense(128, kernel_initializer=initializer)(x)
    output = layers.UnitNormalization()(x) #128

    model = tf.keras.models.Model(input, output)
    optimizer = AdamW()
    model.compile(loss=base_supcon_loss_func,
                  optimizer=optimizer)
    return model
   


base_path = "/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/width_study/width_16"
model_dir = get_next_model_directory(base_path)
model_save_path = os.path.join(model_dir, "model.h5")
plot_save_path = os.path.join(model_dir, "auc_metrics_plot.png")



model = build_flagship_test()
history = model.fit(
    e3_data, e3_labels,
    batch_size = BATCH_SIZE, #69, 129, 989, 2967
    #validation_data = (popstat_data, popstat_labels),
    epochs=num_epochs, verbose=2,
    callbacks = [Metrics(model, nbide_data, nbide_labels),
                #es
                ]
)


model.save(model_save_path)
print(f"Model saved in {model_save_path}")
plot_auc_metrics(history, plot_save_path, period=period, smoothing_window=smoothing_window_size)
