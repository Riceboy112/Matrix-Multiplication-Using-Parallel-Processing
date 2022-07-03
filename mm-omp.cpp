//-----------------------------------------------------------------------
// Matrix Multiplication - OpenMP version to run on shared memory MIMD
//-----------------------------------------------------------------------
//  Written by: Gita Alaghband, Lan Vu 
//  Updated in 10/20/2013
//-----------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <time.h>
#include <omp.h>
#include <cstdlib>
#include <stdio.h>

using namespace std;


//-----------------------------------------------------------------------
//   Get user input of matrix dimension and printing option
//-----------------------------------------------------------------------
bool GetUserInput(int argc, char *argv[],int& n,int& isPrint)
{
	bool isOK = true;

	if(argc < 2) 
	{
		cout << "Arguments:<X> [<Y>]" << endl;
		cout << "X : Matrix size [X x X]" << endl;
		cout << "Y = 1: print the input/output matrix if X < 10" << endl;
		cout << "Y <> 1 or missing: does not print the input/output matrix" << endl;
		isOK = false;
	}
	else 
	{
		//get matrix size
		n = atoi(argv[1]);
		if (n <=0) 
		{
			cout << "Matrix size must be larger than 0" <<endl;
			isOK = false;
		}

		//is print the input/output matrix
		if (argc >=3)
			isPrint = (atoi(argv[2])==1 && n <=9)?1:0;
		else
			isPrint = 0;
	}
	return isOK;
}

//-----------------------------------------------------------------------
//Initialize the value of matrix x[n x n]
//-----------------------------------------------------------------------
void InitializeMatrix(float** &x,int n,float value)
{
	x = new float*[n];
	x[0] = new float[n*n];
    srand (time(NULL));

	for (int i = 1; i < n; i++)	x[i] = x[i-1] + n;

	for (int i = 0 ; i < n ; i++)
	{
		for (int j = 0 ; j < n ; j++)
		{
			
            if (value == 1)  // generate input matrices (a and b)
               x[i][j] = (float)((rand()%10)/(float)2);
            else
                x[i][j] = 0;  // initializing resulting matrix
		}
	}
}
//------------------------------------------------------------------
//Delete matrix x[n x n]
//------------------------------------------------------------------
void DeleteMatrix(float **x,int n)
{
	delete[] x[0];
	delete[] x; 
}
//------------------------------------------------------------------
//Print matrix	
//------------------------------------------------------------------
void PrintMatrix(float **x, int n) 
{
	for (int i = 0 ; i < n ; i++)
	{
		cout<< "Row " << (i+1) << ":\t" ;
		for (int j = 0 ; j < n ; j++)
		{
			printf("%.2f\t", x[i][j]);
		}
		cout<<endl ;
	}
}
//------------------------------------------------------------------
//Do Matrix Multiplication 
//------------------------------------------------------------------
void MultiplyMatrix(float** a, float** b,float** c, int n)
{
    #pragma omp parallel for
    for (int i = 0 ; i < n ; i++)
    {
        for (int k = 0 ; k < n ; k++)
            for (int j = 0 ; j < n ; j++){
                c[i][j] += a[i][k]*b[k][j];
                //if (n <= 4 ){
                //    int tid = omp_get_thread_num();
                //    printf ("thread ID = %d  \t total of therads = %d \n" , tid, omp_get_num_threads()) ;
                //}
            }
        
    }
}
//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[])
{
	float **a,**b,**c;
	int	n,isPrint;
	double runtime;

	if (GetUserInput(argc,argv,n,isPrint)==false) return 1;

    // the code will run according to the number of threads set in the export OMP_NUM_THREADS in the script or environment
	cout << "Matrix multiplication is computed using max of threads = "<< omp_get_max_threads() << " threads or cores" << endl;
    
    cout << " Matrix size  = " << n << endl;
	//Initialize the value of matrix a, b, c
	InitializeMatrix(a,n,1.0);
	InitializeMatrix(b,n,1.0);
	InitializeMatrix(c,n,0.0);

	//Print the input matrices
	if (isPrint==1)
	{
		cout<< "Matrix a[n][n]:" << endl;
		PrintMatrix(a,n); 
		cout<< "Matrix b[n][n]:" << endl;
		PrintMatrix(b,n); 
	}

	runtime = omp_get_wtime();

	MultiplyMatrix(a,b,c,n);

	runtime = omp_get_wtime() - runtime;

	//Print the output matrix
	if (isPrint==1)
	{
		cout<< "Output matrix:" << endl;
		PrintMatrix(c,n); 
	}
	cout<< "Program runs in " << setiosflags(ios::fixed) << setprecision(8) << runtime << " seconds\n";
	
	DeleteMatrix(a,n);	
	DeleteMatrix(b,n);	
	DeleteMatrix(c,n);	
	return 0;
}
