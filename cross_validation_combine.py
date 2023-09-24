# -------------------------
# cross_validation_combine
# -------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import CategoricalCrossentropy
from matplotlib import pyplot as plt 
import make_model
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
from scipy import ndimage

# @tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label
        # data_list_train10.npy
        # label_list_train10.npy
        
        
data_list = [7,8,9,10]
for cv in data_list:
    
    def get_data():
        input_data_list = np.load('data_cross_'+str(cv)+'.npy', allow_pickle=True)
        label_data = np.load('label_cross_'+str(cv)+'.npy', allow_pickle=True)

        # 클래스 수 계산
        num_classes = max(label_data)+ 1
        # 라벨을 원-핫 인코딩으로 변환
        one_hot_labels = np.zeros((len(label_data), num_classes))
        one_hot_labels[np.arange(len(label_data)), label_data] = 1
        #라벨이 1,2,3이여서 차원 1개 더있어서 맨 앞 차원 삭제
        new_arr = np.delete(one_hot_labels, 0, axis=1)
        # print(new_arr)
        
        x_train, x_val, y_train, y_val = train_test_split(input_data_list, new_arr, test_size=0.2, random_state=42)
            
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        # print(validation_loader.shape)

        batch_size = 2
        # Augment the on the fly during training.
        train_dataset = (
            train_loader.shuffle(len(x_train))
            .map(train_preprocessing)
            .batch(batch_size)
            .prefetch(2)
        )

        print(train_dataset)

        # Only rescale.
        validation_dataset = (
            validation_loader.shuffle(len(x_val))
            .map(validation_preprocessing)
            .batch(batch_size)
            .prefetch(2)
        )
        print("데이터 완료")

        # data = train_dataset.take(1)
        # images, labels = list(data)[0]
        # images = images.numpy()
        # image = images[0]
        # print("Dimension of the CT scan is:", image.shape)

        return train_dataset, validation_dataset

    #model = Make_model2.unet(width=64, height=64, depth=None)
    model = make_model.get_model(width=192, height=192, depth=32)
    model.summary()

    train_dataset, validation_dataset = get_data()
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
        filepath = 'best_model_500_'+str(cv)+'.h5', save_best_only=True # ------------------------
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=50)

    # Train the model, doing validation at the end of each epoch

    # print(type(input_data_list), type(label_data))
    #fit
    epochs = 500 # ---------------------------------------------------------------

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        batch_size= 256,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb],
    )

    model.save('model_epoch_500_'+str(cv)+'.h5') # -----------------------------------

    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(["acc", "loss"]):
        ax[i].plot(model.history.history[metric])
        ax[i].plot(model.history.history["val_" + metric])
        ax[i].set_title("Model {}_cv {}".format(metric, 1))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])

    plt.savefig('epoch_500_'+str(cv)+'.png') # ----------------------------------------