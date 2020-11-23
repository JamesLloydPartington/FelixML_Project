#include <stdio.h>
#include <omp.h>
#include <stdbool.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#undef I
complex I = _Complex_I; //Makes it so z = a + b * I

const int nPix = 512; //Size of square
const int FileSize = 841;
const double RScattFacToVolts = 47.913838;
const double complex constPi = 2.0 * 3.1415926535 *  _Complex_I;


int main()
{
  int i, j, n;
  double x, y;
  char Path[] = "/home/jallpa/Documents/GitHub/FelixML_Project/Preprocessing/NextPart";
  char StructFact[] = "/StructureFactor.txt";
  char* concat(char* s1, char* s2);
  char* FilePath = concat(Path, StructFact);


  double complex* Cug = (double complex*)calloc(FileSize, sizeof(double complex));
  int** gVec = (int**)calloc(FileSize, sizeof(int*));
  for(i = 0; i < FileSize; i++)
  {
    gVec[i] = (int*)calloc(3, sizeof(int));
  }

  double** RreUr = (double**)calloc(nPix, sizeof(double*));
  for(i = 0; i < nPix; i++)
  {
    RreUr[i] = (double*)calloc(nPix, sizeof(double));
  }

  void OpenData(char*, double complex*, int**);
  OpenData(FilePath, Cug, gVec);

  double ReDotgVec;
  double complex ExpVar;
  double OneOvernPix = 1.0 / (nPix - 1);

  #pragma omp parallel for default(none) private(i, j, n, x, y, ReDotgVec, ExpVar) shared(OneOvernPix, RreUr, gVec, Cug)
  for(i = 0; i < nPix; i++)
  {
    x = i * OneOvernPix;
    for(j = 0; j < nPix; j++)
    {
      y = j * OneOvernPix;
      for(n = 0; n < FileSize; n++)
      {
        //ReDotgVec = np.dot(Re, gVectors[n])
        ReDotgVec = x * gVec[n][0] + y * gVec[n][1];
        ExpVar = cexp(constPi * ReDotgVec);
        RreUr[j][i]+= creal(Cug[n] * ExpVar);
      }

    }
  }
  for(i = 0; i < nPix; i++)
  {
    for(j = 0; j < nPix; j++)
    {
      RreUr[j][i] = RreUr[j][i] * RScattFacToVolts;
      //printf("%g ", RreUr[j][i]);
    }
    //printf("\n");
  }


  return(0);
}

//void Dot3(double* Result; double* )


void OpenData(char* NameofFile, double complex* Cug, int** gVec)
{
  FILE * file_ptr = fopen(NameofFile, "r"); //writing to a file called Frac1
  int i;
  double A, B, C, D;
  int E, F, G;
  //printf("%s\n ", NameofFile);
  for(i = 0; i < FileSize; i++)
  {
    fscanf(file_ptr, "\t%d\t%d\t%d\t%lf\t%lf\t%lf\t%lf", &gVec[i][0], &gVec[i][1], &gVec[i][2], &A, &B, &C, &D);
    Cug[i] = C + I * D;
    //printf("%g + i%g\n", creal(Cug[i]), cimag(Cug[i]));
    //printf("%d %d %d\n", gVec[i][0], gVec[i][1], gVec[i][2]);
  }
}




char* concat(char* s1, char* s2)
{
    char *result = malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}
