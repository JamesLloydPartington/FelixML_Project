import numpy as np
import os


Path = "/media/ug-ml/Samsung_T5/training3000/1to1000_all_directions"
NewPath = "/media/ug-ml/Samsung_T5/training3000/AllDirectionDataSet"

NameNumber = 0 #for 1-1000 = 0. For 1001-2000 = 1000 ect

Original = np.load(Path+"/K12874Si9696Al9504O384_0050_+0+0+0.npy")
New = np.load(NewPath+"/ChunkImage_0.npy")

print(Original.shape)
print(New.shape)
