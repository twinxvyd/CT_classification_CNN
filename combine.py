import make_model
from tensorflow import keras
from matplotlib import pyplot as plt 
from tensorflow.keras.losses import CategoricalCrossentropy
import get_data
import numpy as np
import tensorflow as tf


#model = Make_model2.unet(width=64, height=64, depth=None)
model = make_model.get_model(width=512, height=512, depth=64)
model.summary()

train_dataset, validation_dataset = get_data.get_data()
initial_learning_rate = 0.00001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.96, staircase=True
)
model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)


# Define callbacks.
#https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath = "best_model_300.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=50)

# Train the model, doing validation at the end of each epoch

# print(type(input_data_list), type(label_data))
#fit
epochs = 500

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    batch_size= 256,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb],
)

model.save('model_epoch_300.h5')

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

plt.savefig('epoch_300.png')