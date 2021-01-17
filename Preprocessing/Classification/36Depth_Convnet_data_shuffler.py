import numpy as np
rng = np.random.default_rng()

# open data and labels
source_dir = "/media/ug-ml/Samsung_T5/training3000/1to1000_all_directions_200_sample"
arr_data = np.load(source_dir+"/ImageData.npy")
arr_labels = np.load(source_dir+"/ImageLab.npy")

# create a set of indexes for training and a set for validation.
number_of_crystals = np.size(arr_data, 0)
validation_fraction = 0.5
number_of_validation_crystals = int(number_of_crystals*validation_fraction)
number_of_training_crystals = number_of_crystals - number_of_validation_crystals
indexes = np.arange(0, number_of_crystals)
rng.shuffle(indexes)
train_indexes = indexes[number_of_validation_crystals:]
validation_indexes = indexes[:number_of_validation_crystals]

# create sets of training data and validation data
train_data = arr_data[train_indexes].reshape((number_of_training_crystals*10,36,128,128))
tmp_train_labels = arr_labels[train_indexes].reshape((number_of_training_crystals*10,36,1))
train_labels = np.array([i for j in tmp_train_labels for i in j[0] ])
validation_data = arr_data[validation_indexes].reshape((number_of_validation_crystals*10,36,128,128))
tmp_validation_labels = arr_labels[validation_indexes].reshape((number_of_validation_crystals*10,36,1))
validation_labels = np.array([i for j in tmp_validation_labels for i in j[0] ])
print(validation_labels.shape)
shuffle_train_indexes = np.arange(0, number_of_training_crystals*10)
shuffle_validation_indexes = np.arange(0, number_of_validation_crystals*10)
rng.shuffle(shuffle_train_indexes)
rng.shuffle(shuffle_validation_indexes)

train_data = train_data[shuffle_train_indexes]
train_labels = train_labels[shuffle_train_indexes]
validation_data = validation_data[validation_indexes]
validation_labels = validation_labels[validation_indexes]
np.save("/media/ug-ml/Samsung_T5/training3000/1to1000_all_directions_200_sample/train_data.npy", train_data)
np.save("/media/ug-ml/Samsung_T5/training3000/1to1000_all_directions_200_sample/train_labels.npy", train_labels)
np.save("/media/ug-ml/Samsung_T5/training3000/1to1000_all_directions_200_sample/validation_data.npy", validation_data)
np.save("/media/ug-ml/Samsung_T5/training3000/1to1000_all_directions_200_sample/validation_labels.npy", validation_labels)

print(train_data)
