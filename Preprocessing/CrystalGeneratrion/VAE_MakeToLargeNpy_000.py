import os
import numpy as np
from shutil import copyfile


def NumberInEachSet(Path):
	CrystalsInSet = [] #number of crystals in Test, Training and Validation data respectivly
	Path_i = sorted(os.listdir(Path))
	index_i = 0
	for i in Path_i: # i is the Test, training and validation
		CrystalsInSet.append(0)
		Path_j = sorted(os.listdir(Path + "/" + i))
		index_j = 0
		for j in Path_j: #j is the crystal
			index_j+=1
		CrystalsInSet[index_i] = index_j
		index_i+=1
	return CrystalsInSet


def AllDataToArray(Path, SaveLocation, CrystalsInSet):
	if not os.path.exists(SaveLocation):
            os.mkdir(SaveLocation)
	AllData = []
	for i in CrystalsInSet:
		ImageArray = np.zeros(i * 128 * 128, dtype = np.float32).reshape(i, 128, 128)
		AllData.append(ImageArray)

	Path_i = sorted(os.listdir(Path))
	index_i = 0
	for i in Path_i: # i is the Test, training and validation
		Path_j = sorted(os.listdir(Path + "/" + i))
		index_j = 0
		for j in Path_j: #j is the crystal
			Path_k = Path + "/" + i + "/" + j + "/Output.npy"
			data = np.load(Path_k)
			AllData[index_i][index_j] = data
			index_j+=1
		np.save(SaveLocation + "/" + i + "_Output.npy", AllData[index_i])
		index_i+=1

	return AllData



Path = "/home/physics/phupvw/felixML/VAE_000_1/Data"
SaveLocation = "/home/physics/phupvw/felixML/VAE_000_1/NpyFile"
CrystalsInSet = NumberInEachSet(Path)
print("Number in each set: ", CrystalsInSet)

AllData = AllDataToArray(Path, SaveLocation, CrystalsInSet)
