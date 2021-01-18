#REMOVE FILE 2017 FROM DATA
import os
import numpy as np
from shutil import copyfile

def CreateDir(NewDataPath):
    Data = "/Data"
    CifFlolder = "/CifFlolder"
    StructureFolder = "/StructureFactors"
    SubFolders = [Data, CifFlolder, StructureFolder]
    TypeOfData = ["/Training", "/Validation", "/Test"]

    for i in SubFolders:
        Path_i = NewDataPath + i
        if not os.path.exists(Path_i):
            os.mkdir(Path_i)

    for i in TypeOfData:
        Path_i = NewDataPath + Data + i
        if not os.path.exists(Path_i):
            os.mkdir(Path_i)

    AllFolders = [SubFolders, TypeOfData]
    return(AllFolders)

def WorkingCrystalIndex(Path):
	WorkingCrystalIndex = []
	Path_i = sorted(os.listdir(Path))
	for i in Path_i: #i is which crystal
		Path_j = sorted((os.listdir(Path + "/" + i)))
		if(len(Path_j) == 3): #If there are only 3 files, that means the simulation did not run
			print("The simulation failed for crystal: ", i)
		else:
			for j in Path_j: #j is the .hkl, .inp, .cif files AND the folder containing the simulated data (expect 4)
				if(j[-4:] == ".inp" or j[-4:] == ".hkl" or j[-4:] == ".cif" or j[-7:] == "128x128"):
					#Do nothing with files
					nothing = 1
				else:
					print("There is an unexpected file called: ", Path + "/" + i + "/" + j)
					return(0)
			WorkingCrystalIndex.append(int(i))
	return(WorkingCrystalIndex)

def ShuffleIndexCreator(WorkingCrystalIndex):
    rng = np.random.default_rng()
    length = len(WorkingCrystalIndex)
    ShuffleIndex = np.arange(length, dtype = np.int)
    rng.shuffle(ShuffleIndex)
    print("Number of working crystals: ", length)
    return(ShuffleIndex)

def BinToDirectory(RawDataPath, NewDataPath, AllFolders, ShuffleIndex, DataRatios, File_CProgram):
	SubFolders = AllFolders[0] # Data, CifFolder, StructureFile
	TypeOfData = AllFolders[1] # Training, Validation, Test
	#Classes = AllFolders[2] #0, 1, 2 ... 9 No classes in VAE

	Iteration = 0
	NumberOfCrystals = len(ShuffleIndex)

	NumberTraining = int(NumberOfCrystals * DataRatios[0])
	NumberValidation = int(NumberOfCrystals * DataRatios[1])
	NumberTest = NumberOfCrystals - NumberTraining - NumberValidation

	WhichData = 0 # 0 for training, 1 for validation, 2 for test

	Path_i = sorted(os.listdir(RawDataPath))
	for i in Path_i: #i is crystal
		Path_j = sorted((os.listdir(RawDataPath + "/" + i)))
		if(len(Path_j) != 3): #If there are only 3 files, that means the simulation did not run
			CrystalNo = int(i)
			RandomIndex = ShuffleIndex[Iteration]
			if(RandomIndex < NumberTraining):
				WhichData = 0 #Training
			elif(RandomIndex < NumberTraining + NumberValidation):
				WhichData = 1 #Validation
			else:
				WhichData = 2 #Test
			print("Crystal", int(i))
			#print("Index", Iteration)
			for j in Path_j: #j is the .hkl, .inp, .cif files AND the folder containing the simulated data (expect 4)
				if(j[-4:] == ".inp" or j[-4:] == ".hkl" or j[-4:] == ".cif"):
					#Do nothing with files
					nothing = 1
				elif(j[-7:] == "128x128"):
					Path_k = sorted((os.listdir(Path + "/" + i + "/" + j)))
					for k in Path_k: #k are the .cif, .txt and .bin files
						if(k[-4:] == ".cif"):
							#CRYSTAL FILE INFOMATION
							RawCifFile = RawDataPath + "/" + i + "/" + j + "/" + k
							CopyCif(RawCifFile, NewDataPath, AllFolders, CrystalNo)
						elif(k[-4:] == ".bin"):
							if(k[-10:] == "+0+0+0.bin"): #Select only the central beam bin file
								BinFile_000 = RawDataPath + "/" + i + "/" + j + "/" + k
								BinToNpy(BinFile_000, NewDataPath, CrystalNo, WhichData, File_CProgram)
						elif(k[-4:] == ".txt"):
							RawStructureFile = RawDataPath + "/" + i + "/" + j + "/" + k
							CopyStructureFactor(RawStructureFile, NewDataPath, AllFolders, CrystalNo)
						else:
							print("File or folder with name (Warning1):", k)
				else:
					print("There is an unexpected file called: ", j)
					return(0)
			Iteration+=1
	return

def CopyCif(RawCifFile, NewDataPath, AllFolders, CrystalNo):
    SubFolders = AllFolders[0] # Data, CifFolder, StructureFile
    #TypeOfData = AllFolders[1] # Training, Validation, Test
    #Classes = AllFolders[2] #0, 1, 2 ... 9 No classes in VAE
    NewCifFile = NewDataPath + SubFolders[1] + "/" + str(CrystalNo) + ".cif"
    copyfile(RawCifFile, NewCifFile)

def BinToNpy(BinFile_000, NewDataPath, CrystalNo, WhichData, File_CProgram):
    SubFolders = AllFolders[0] # Data, CifFolder, StructureFile
    TypeOfData = AllFolders[1] # Training, Validation, Test
    #Classes = AllFolders[2] #0, 1, 2 ... 9

    data = np.fromfile(BinFile_000, dtype=np.float64).reshape(128, 128)
    Image_000 = data.astype(np.float32)

    if(WhichData == 0): #Training
        FolderLocation = NewDataPath + SubFolders[0] + TypeOfData[0] + "/" + str(CrystalNo) + "/"
    elif(WhichData == 1): #Validation
        FolderLocation = NewDataPath + SubFolders[0] + TypeOfData[1] + "/" + str(CrystalNo) + "/"
    else: #Test
        FolderLocation = NewDataPath + SubFolders[0] + TypeOfData[2] + "/" + str(CrystalNo) + "/"

    if not os.path.exists(FolderLocation):
            os.mkdir(FolderLocation)
    np.save(FolderLocation + "Output.npy", Image_000)

    PotentialDist_Location = FolderLocation + "Input.txt" #Where the unit cell potential will be stored
    Structure_Location = NewDataPath + SubFolders[2] + "/" + str(CrystalNo) + ".txt" #Where the structure file is, this will be used to make the potential of the unit cell

    File_CProgram.write(PotentialDist_Location + " " + Structure_Location + "\n") #The C program that calculates the unit cell potential needs to read these 2 file locations
    return

def CopyStructureFactor(RawStructureFile, NewDataPath, AllFolders, CrystalNo):
    SubFolders = AllFolders[0] # Data, CifFolder, StructureFile
    #TypeOfData = AllFolders[1] # Training, Validation, Test
    #Classes = AllFolders[2] #0, 1, 2 ... 9 No classes in VAE
    NewStructureFile = NewDataPath + SubFolders[2] + "/" + str(CrystalNo) + ".txt"
    copyfile(RawStructureFile, NewStructureFile)
    return


Path = "/home/physics/phupvw/felixML/VAE_ALL"
NewPath = "/home/physics/phupvw/felixML/VAE_000_1" #This folder will be created and will be where all the data will go

TrainingRatio = 0.85
ValidationRatio = 0.1
TestRatio = 1 - TrainingRatio - ValidationRatio
DataRatios = [TrainingRatio, ValidationRatio, TestRatio]

AllFolders = CreateDir(NewPath)
SubFolders = AllFolders[0] # Data, CifFolder, StructureFile
TypeOfData = AllFolders[1] # Training, Validation, Test
#Classes = AllFolders[2] #0, 1, 2 ... 9 No classes in VAE

WorkingCrystalIndex = WorkingCrystalIndex(Path)
ShuffleIndex = ShuffleIndexCreator(WorkingCrystalIndex)

File_CProgram  = open(NewPath + SubFolders[2] +"/FilePaths.txt", "a")

BinToDirectory(Path, NewPath, AllFolders, ShuffleIndex, DataRatios, File_CProgram)

File_CProgram.close()
