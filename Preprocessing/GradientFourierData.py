#echo 1 | sudo tee /proc/sys/vm/overcommit_memory
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

import numpy as np
import matplotlib.pyplot as plt

Path = "Quarter32Bit"

AllData = np.load(Path + "/ImageDataNorm.npy")
AllLabel = np.load(Path + "/ImageLab.npy")


def Gradient(Array):
	ArrayCopy = Array.copy()

	for i in range(0, Array.shape[0]):
		for j in range(0, Array.shape[1]): #Boundary conditions
			if i == 0:
				if j == 0:
					ArrayCopy[i][j] = ((Array[i][j] - Array[i + 1][j]) ** 2) + ((Array[i][j] - Array[i][j + 1]) ** 2)
					ArrayCopy[i][j] = (ArrayCopy[i][j] / 2) ** 0.5
				elif j == Array.shape[1] - 1:
					ArrayCopy[i][j] = ((Array[i][j] - Array[i + 1][j]) ** 2) + ((Array[i][j] - Array[i][j - 1]) ** 2)
					ArrayCopy[i][j] = (ArrayCopy[i][j] / 2) ** 0.5
				else:
					ArrayCopy[i][j] = ((Array[i][j] - Array[i + 1][j]) ** 2) + ((Array[i][j] - Array[i][j + 1]) ** 2) + ((Array[i][j] - Array[i][j - 1]) ** 2)
					ArrayCopy[i][j] = (ArrayCopy[i][j] / 3) ** 0.5
			elif i == Array.shape[0] - 1:
				if j == 0:
					ArrayCopy[i][j] = ((Array[i][j] - Array[i - 1][j]) ** 2) + ((Array[i][j] - Array[i][j + 1]) ** 2)
					ArrayCopy[i][j] = (ArrayCopy[i][j] / 2) ** 0.5
				elif j == Array.shape[1] - 1:
					ArrayCopy[i][j] = ((Array[i][j] - Array[i - 1][j]) ** 2) + ((Array[i][j] - Array[i][j - 1]) ** 2)
					ArrayCopy[i][j] = (ArrayCopy[i][j] / 2) ** 0.5
				else:
					ArrayCopy[i][j] = ((Array[i][j] - Array[i - 1][j]) ** 2) + ((Array[i][j] - Array[i][j + 1]) ** 2) + ((Array[i][j] - Array[i][j - 1]) ** 2)
					ArrayCopy[i][j] = (ArrayCopy[i][j] / 3) ** 0.5
			elif j == 0:
				if i != 0 and i != Array.shape[0] - 1:
					ArrayCopy[i][j] = ((Array[i][j] - Array[i + 1][j]) ** 2) + ((Array[i][j] - Array[i - 1][j]) ** 2) + ((Array[i][j] - Array[i][j + 1]) ** 2)
					ArrayCopy[i][j] = (ArrayCopy[i][j] / 3) ** 0.5
			elif j == Array.shape[1] - 1:
				if i != 0 and i != Array.shape[0] - 1:
					ArrayCopy[i][j] = ((Array[i][j] - Array[i + 1][j]) ** 2) + ((Array[i][j] - Array[i - 1][j]) ** 2) + ((Array[i][j] - Array[i][j - 1]) ** 2)
					ArrayCopy[i][j] = (ArrayCopy[i][j] / 3) ** 0.5

			else:
				ArrayCopy[i][j] = ((Array[i][j] - Array[i + 1][j]) ** 2) + ((Array[i][j] - Array[i - 1][j]) ** 2) + ((Array[i][j] - Array[i][j + 1]) ** 2) + ((Array[i][j] - Array[i][j - 1]) ** 2)
				ArrayCopy[i][j] = (ArrayCopy[i][j] / 4) ** 0.5

	return(ArrayCopy)

def Fourier(Array):
	FArray = np.fft.fft2(Array)
	FArray[0][0] = FArray[1][1] #In the FT the (0, 0) pixel is always much brighter than all the others, so set it to the next diagonal pixel
	return(FArray)

def Norm(AllArray):
	n = 0
	for i in AllArray:
		for j in i:
			min = j[0][0]
			max = j[0][0]
			for k in j:
				for l in k:
					if(l > max):
						max = l
					if(l < min):
						min = l
			j = (j - min) / (max - min)
		print(n)
		n = n + 1
	return AllArray





DiIm = AllData.shape
Di3Im = (DiIm[0], DiIm[1], DiIm[2], DiIm[3], 3)
print(Di3Im)

GradArray = AllData.copy()
FourierArray = AllData.copy()
ThreeDArray = np.arange((DiIm[0] * DiIm[1] * DiIm[2] * DiIm[3] * 3), dtype = np.float32).reshape(Di3Im)
print("Arrays created")

for i in range(0, AllData.shape[0]):
	for j in range(0, AllData.shape[1]):
		GradArray[i][j] = Gradient(GradArray[i][j])
		FourierArray[i][j] = Fourier(FourierArray[i][j])
	print("Stage 1", i)

print("Values generated")

GradArray = Norm(GradArray)
FourierArray = Norm(FourierArray)

print("Values Normalised")

for i in range(0, AllData.shape[0]):
	for j in range(0, AllData.shape[1]):
		for k in range(0, AllData.shape[2]):
			for l in range(0, AllData.shape[3]):
				ThreeDArray[i][j][k][l][0] = AllData[i][j][k][l]
				ThreeDArray[i][j][k][l][1] = GradArray[i][j][k][l]
				ThreeDArray[i][j][k][l][2] = FourierArray[i][j][k][l]
	print("Stage 2", i)

print("3D array completed")

np.save(Path+"/ImageGradFourierNorm.npy", ThreeDArray)

print("Saved")

col = 5
row = 2
for i in range(0, ThreeDArray.shape[0]):
	fig=plt.figure(figsize = (64, 64))
	for j in range(0, ThreeDArray.shape[1]):
		img = ThreeDArray[i][j][:, :, :]
		fig.add_subplot(row, col, j + 1)
		plt.imshow(img)
	print(i)
	plt.show()
