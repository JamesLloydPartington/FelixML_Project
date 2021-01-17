import numpy as np
import os


Path = "F:/training3000/1to1000_all_directions"
NewPath = "F:/training3000/1to1000_all_directions_200_sample"


def ThicknessString(String): #Takes the file name string, looks at the thickness part and returns a integer corresponding to the thickness class
    if(String == "0050"):
        return 0
    elif(String == "0250"):
        return 1
    elif(String == "0450"):
        return 2
    elif(String == "0650"):
        return 3
    elif(String == "0850"):
        return 4
    elif(String == "1050"):
        return 5
    elif(String == "1250"):
        return 6
    elif(String == "1450"):
        return 7
    elif(String == "1650"):
        return 8
    elif(String == "1850"):
        return 9
    else: #If something goes wrong, an error will be printed
        print("ERROR")
        return 10

def Position():
    IndexArray = np.zeros(8 * 8, dtype = int).reshape(8, 8) #Declare Image array
    index = 0
    for i in range(0, 8):
        for j in range(0, 8):
            if(i >= j):
                IndexArray[i][j] = index
                index+=1
    return IndexArray

DirectionIndex = Position()

Name = ""
Thickness = ""
Direction = ""
DirectionArray = [0, 0 ,0]
CrystalNo = 0
k_start = 0
k_end = 0
k = 0
Direction_n = 0
fileNo = 0
Directon_i = 0
thickness_i = 0

File1  = open(NewPath +"/Key.txt", "w+") #This file will contain two integers and a name of the original file which can be


imageName = sorted(os.listdir(Path)) #Sorts the lists of all the crysrals in alphabetical order, and will also order the thicknesses.
#print("Sorted")

N = 200
Dimension = [N, 10, 36, 128, 128]
size = Dimension[0] * Dimension[1] * Dimension[2] * Dimension[3] * Dimension[4]
AllData = np.arange(size, dtype = np.float32).reshape(Dimension) #Declare Image array
AllLabel = np.arange(Dimension[0] * Dimension[1] * Dimension[2], dtype = int).reshape(Dimension[0], Dimension[1], Dimension[2])


fileNo0 = 0
savename = imageName[0].split("_")[0]
for i in imageName:
    #print(i)
    Name = i.split("_")[0]
    if(Name != savename):
        CrystalNo+=1
        savename = Name
        fileNo0 = fileNo
        if(CrystalNo >= N):
            break
    Thickness = i.split("_")[1]
    thickness_i = ThicknessString(Thickness)
    Direction = list(map(int, i.split("_")[2][:-4].split("+")[1:]))
    #print(Direction)
    if(Direction[0] <= 14 and Direction[1] <=14):
        Directon_i = DirectionIndex[int(Direction[0]/2)][int(Direction[1]/2)]

    fileNo = fileNo + 1
    data = np.load(Path + "/" + i) #Open image data
    #print(data)
    AllData[CrystalNo][thickness_i][Directon_i] = data
    AllLabel[CrystalNo][thickness_i][Directon_i] = thickness_i
    File1.write(str(CrystalNo)+" "+ str(thickness_i) +" " + str(Direction) + " " + i +"\n")

    #print(Name, Thickness, Direction)


#print(fileNo)
#print(CrystalNo)
#print(AllData)
#print(AllLabel)

File1.close()
np.save(NewPath+"/ImageData.npy", AllData) #The image data is saved as ImageData.npy
np.save(NewPath+"/ImageLab.npy", AllLabel) #The label data is saved as AllLabel.npy and the Key is was saved as Key.txt


print("Number of images:", fileNo) #Check to see how many images there are
