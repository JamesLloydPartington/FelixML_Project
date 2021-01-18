import os
import numpy as np
from shutil import copyfile

Path = "/home/physics/phupvw/felixML/VAE_000_1/Data"
Path_i = sorted(os.listdir(Path)) #Training Validation and Test
for i in Path_i: #i training, validation and test
    Path_j = sorted(os.listdir(Path + "/" + i))
    for j in Path_j: #j is the crystal
        InputFileName = Path + "/" + i + "/" + j +"/" + "Input.txt"
        InputNpyFileName = Path + "/" + i + "/" + j +"/" + "Input.npy"
        Image = np.loadtxt(InputFileName).astype(np.float32)
        np.save(InputNpyFileName, Image)
        os.remove(InputFileName)
