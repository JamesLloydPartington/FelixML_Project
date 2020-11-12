import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.utils import to_categorical
#from keras.models import load_model
import os
import matplotlib.pyplot as plt
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

PixelDimension = 36

# open data and labels
source_dir = "/media/ug-ml/Samsung_T5/training3000/1to1000_all_directions_200_sample"

train_images = np.load("/media/ug-ml/Samsung_T5/training3000/1to1000_all_directions_200_sample/train_data.npy")

train_lab = to_categorical(np.load("/media/ug-ml/Samsung_T5/training3000/1to1000_all_directions_200_sample/train_labels.npy"))

val_images = np.load("/media/ug-ml/Samsung_T5/training3000/1to1000_all_directions_200_sample/validation_data.npy")

val_lab = to_categorical(np.load("/media/ug-ml/Samsung_T5/training3000/1to1000_all_directions_200_sample/validation_labels.npy"))

print(train_images.shape,"\n\n\n\n\n\n\n\n\n\n")

model = models.Sequential()
model.add(layers.SeparableConv2D(128, (3, 3), activation='relu', data_format='channels_first', input_shape=(PixelDimension, 128, 128)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.SeparableConv2D(128, (3, 3), data_format='channels_first', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.SeparableConv2D(128, (3, 3), data_format='channels_first',  activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(512, activation='relu', kernel_regularizer = l2(0.0001)))
model.add(layers.Dense(10, activation='softmax', kernel_regularizer = l2(0.0001)))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate = 0.0005), metrics=['acc'])

#model.summary()


EarlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
Checkpoint = ModelCheckpoint("3DCovBest.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

#model = models.load_model('F:\\training3000\\pngs_thicknesses_model_L2.h5')
#model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['acc'])


history = model.fit(train_images, train_lab, epochs=100, batch_size = 64, validation_data = (val_images, val_lab), shuffle = True, callbacks=[EarlyStop, Checkpoint])

model.save('3DCovLast.h5')

test_loss, test_acc = model.evaluate(test_images, test_lab)

with open('3DCovPICKLE', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
