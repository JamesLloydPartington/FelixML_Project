import numpy as np
import os
import matplotlib.pyplot as plt
import math
import cmath

#print u'\u212B'.encode('utf-8')
print u'你'.encode('utf-8')

Path = "/home/jallpa/Documents/GitHub/FelixML_Project/Preprocessing/NextPart"
StructFact = "/StructureFactor.txt"

PathOfFile = Path + StructFact
with open(PathOfFile) as textFile:
    lines = [line.split() for line in textFile]

FileLength = len(lines)

gVectors = np.zeros(FileLength * 3).reshape(FileLength, 3)
Ug = np.zeros(FileLength * 2, dtype = np.cdouble).reshape(FileLength, 2)
for i in range(0, FileLength):
    gVectors[i][0] = int(lines[i][0])
    gVectors[i][1] = int(lines[i][1])
    gVectors[i][2] = int(lines[i][2])
    Ug[i][0] = float(lines[i][5])
    Ug[i][1] = float(lines[i][6])

#print(Ug)

nPix = 128

Vx = np.array([1, 0, 0])
Vy = np.array([0, 1, 0])
MagX = 1
MagY = 1

x = np.arange(nPix, dtype = np.float) / (nPix - 1)
y = np.arange(nPix, dtype = np.float) / (nPix - 1)

RScattFacToVolts = 47.913838
constPi = 2 * 3.1415926535 * 1j

RreUr = np.zeros(nPix * nPix).reshape(nPix, nPix)
RimUr = np.zeros(nPix * nPix).reshape(nPix, nPix)

Re = [0, 0, 0]
for i in range(0, nPix):
    for j in range(0, nPix):
        Re = x[i] * Vx + y[j] * Vy
        #print(Re)
        for n in range(0, FileLength):
            cug = Ug[n][0] + 1j * Ug[n][1]

            ReDotgVec = np.dot(Re, gVectors[n])

            ExpVar = cmath.exp(constPi * ReDotgVec)
            #print(ReDotgVec)
            RreUr[j][i]+= (cug * ExpVar).real
            #RimUr[j][i]+=  (1j * cug * ExpVar).imag



RreUr = RreUr * RScattFacToVolts
#RimUr = RimUr * RScattFacToVolts
print(RreUr)


fig=plt.figure(figsize = (32, 32))
img = RreUr
plt.imshow(img, cmap = "gray")
plt.legend()
plt.show()
