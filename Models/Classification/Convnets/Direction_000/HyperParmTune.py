#echo 1 | sudo tee /proc/sys/vm/overcommit_memory
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

import numpy as np

Path = "Quarter32Bit"

AllData = np.load(Path + "/ImageDataNorm.npy")
AllLabel = np.load(Path + "/ImageLab.npy")

No_Crystals = AllData.shape[0]
Total_ints = AllData.shape[0] * AllData.shape[1] * AllData.shape[2] * AllData.shape[3] 

Shuffle_Index = np.arange(No_Crystals)
np.random.shuffle(Shuffle_Index)

###########
TrainRatio = 0.8
ValidationRatio = 0.1 ## A fraction of the Traning data
TestRatio = 1 - TrainRatio - ValidationRatio

###########

TrainCrystalNo = round(TrainRatio * No_Crystals)
ValCrystalNo = round(ValidationRatio * No_Crystals)
TestCrystalNo = No_Crystals - TrainCrystalNo - ValCrystalNo
PixleNo = AllData.shape[2] * AllData.shape[3]

train_images = np.zeros(TrainCrystalNo * PixleNo * AllData.shape[1], dtype = np.float32).reshape(TrainCrystalNo, AllData.shape[1], AllData.shape[2], AllData.shape[3])
val_images = np.zeros(ValCrystalNo * PixleNo * AllData.shape[1], dtype = np.float32).reshape(ValCrystalNo, AllData.shape[1], AllData.shape[2], AllData.shape[3])
test_images = np.zeros(TestCrystalNo * PixleNo * AllData.shape[1], dtype = np.float32).reshape(TestCrystalNo, AllData.shape[1], AllData.shape[2], AllData.shape[3])

train_lab = np.zeros(TrainCrystalNo * AllData.shape[1], dtype = np.uint8).reshape(TrainCrystalNo, AllData.shape[1])
val_lab = np.zeros(ValCrystalNo * AllData.shape[1], dtype = np.uint8).reshape(ValCrystalNo, AllData.shape[1])
test_lab = np.zeros(TestCrystalNo * AllData.shape[1], dtype = np.uint8).reshape(TestCrystalNo, AllData.shape[1])

for i in range(0, No_Crystals): #Put Shuffled Crystals into training and validation and test
	if(i < TrainCrystalNo):
		train_images[i] = AllData[Shuffle_Index[i]]
		train_lab[i] = AllLabel[Shuffle_Index[i]]

	elif(i < TrainCrystalNo + ValCrystalNo):
		val_images[i - TrainCrystalNo] = AllData[Shuffle_Index[i]]
		val_lab[i - TrainCrystalNo] = AllLabel[Shuffle_Index[i]]	
	
	else:
		test_images[i - TrainCrystalNo - ValCrystalNo] = AllData[Shuffle_Index[i]]
		test_lab[i - TrainCrystalNo - ValCrystalNo] = AllLabel[Shuffle_Index[i]]
		

Shuffle_Index_Train = np.arange(TrainCrystalNo * AllData.shape[1])
Shuffle_Index_Val = np.arange(ValCrystalNo * AllData.shape[1])
Shuffle_Index_Test = np.arange(TestCrystalNo * AllData.shape[1])

np.random.shuffle(Shuffle_Index_Train)
np.random.shuffle(Shuffle_Index_Val)
np.random.shuffle(Shuffle_Index_Test)

train_images = train_images.reshape(-1, AllData.shape[2], AllData.shape[3], 1)
val_images = val_images.reshape(-1, AllData.shape[2], AllData.shape[3], 1)
test_images = test_images.reshape(-1, AllData.shape[2], AllData.shape[3], 1)
train_lab = train_lab.reshape(-1)
val_lab = val_lab.reshape(-1)
test_lab = test_lab.reshape(-1)

train_images_copy = train_images
val_images_copy = val_images
test_images_copy = test_images
train_lab_copy = train_lab
val_lab_copy = val_lab
test_lab_copy = test_lab


for i in range(0,TrainCrystalNo * AllData.shape[1]):
	train_images_copy[i] = train_images[Shuffle_Index_Train[i]]
	train_lab_copy[i] = train_lab[Shuffle_Index_Train[i]]
for i in range(0,ValCrystalNo * AllData.shape[1]):
	val_images_copy[i] = val_images[Shuffle_Index_Val[i]]
	val_lab_copy[i] = val_lab[Shuffle_Index_Val[i]]
for i in range(0,TestCrystalNo * AllData.shape[1]):
	test_images_copy[i] = test_images[Shuffle_Index_Test[i]]
	test_lab_copy[i] = test_lab[Shuffle_Index_Test[i]]

train_images = train_images_copy.astype("float32")
val_images = val_images_copy.astype("float32")
test_images = test_images_copy.astype("float32")
train_lab = train_lab_copy
val_lab = val_lab_copy
test_lab = test_lab_copy



####################################################################################

from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
#from keras.models import load_model
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3" 


def create_model(optimizer, loss, regularizers):
	model = models.Sequential()
	model.add(layers.SeparableConv2D(128, (3, 3), activation='relu', input_shape=(64, 64, 1)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.SeparableConv2D(128, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.SeparableConv2D(128, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Flatten())
	model.add(layers.Dropout(0.36))
	model.add(layers.Dense(512, activation='relu', kernel_regularizer = regularizers))
	model.add(layers.Dense(10, activation='softmax', kernel_regularizer = regularizers))

	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
	return model


model = KerasClassifier(build_fn=create_model, verbose = 2)
optimizers = ["adam", "rmsprop"]
epochs = [50]
batches = [32]
loss = ["categorical_crossentropy"]
regularizers = [l2(0.01), l2(0.001), l2(0.0001)]

param_grid = dict(optimizer = optimizers, epochs = epochs, batch_size = batches, loss = loss, regularizers = regularizers)
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3)

grid_result = grid.fit(train_images, train_lab, validation_data = (val_images, val_lab))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))




