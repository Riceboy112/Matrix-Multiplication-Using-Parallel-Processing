/******************************************************************************
* FILE: mpi_mm.c
* DESCRIPTION:  
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks-1 worker tasks.
*   NOTE:  C and Fortran versions of this code differ because of the way
*   arrays are stored/passed.  C arrays are row-major order but Fortran
*   arrays are column-major order.
* AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
*   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
* LAST REVISED: 04/13/05
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

using namespace std;


//-----------------------------------------------------------------------
//   Get user input for matrix dimension or printing option
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
// allocate memory for the 2D matrices [N x N]
//-----------------------------------------------------------------------
double **alloc_2d_double(int n) {
    double *data = (double *)malloc(n * n * sizeof(double));
    double **array= (double **)malloc(n*sizeof(double*));
    for (int i=0; i<n; i++)
        array[i] = &(data[n*i]);
    
    return array;
}



int main (int argc, char *argv[])
{
int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	rows,                  /* rows of matrix A sent to each worker */
	averow, extra, offset, /* used to determine rows sent to each worker */
	i, j, k, rc;           /* misc */

    double **a, **b, **c;
    double runtime;
    
    
    int N =0, isPrint;
    
    if (GetUserInput(argc,argv,N,isPrint)==false) return 1;

    a = alloc_2d_double(N);
    b = alloc_2d_double(N);
    c = alloc_2d_double(N);
    
    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    if (numtasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    
    numworkers = numtasks-1;


/**************************** master task ************************************/
   if (taskid == MASTER)
   {
      printf("mpi_mm has started with %d tasks.\n",numtasks);
      printf("Initializing arrays...\n");
      for (i=0; i<N; i++)
         for (j=0; j<N; j++)
            a[i][j]= i+j;
      for (i=0; i<N; i++)
         for (j=0; j<N; j++)
            b[i][j]= i*j;
       
       //Get start time
       runtime = MPI_Wtime();

      /* Send matrix data to the worker tasks */
      averow = N/numworkers;
      extra = N%numworkers;
      offset = 0;
      mtype = FROM_MASTER;
      for (dest=1; dest<=numworkers; dest++)
      {
         rows = (dest <= extra) ? averow+1 : averow;   	
         //printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
         MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&(a[offset][0]), rows*N, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&(b[0][0]), N*N, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
         offset = offset + rows;
      }

      /* Receive results from worker tasks */
      mtype = FROM_WORKER;
      for (i=1; i<=numworkers; i++)
      {
         source = i;
         MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&(c[offset][0]), rows*N, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
        // printf("Received results from task %d\n",source);
      }
       
       runtime = MPI_Wtime() - runtime;

      /* Print results */
       /*
       if (isPrint == 1) { // show the result
           printf("******************************************************\n");
           printf("Result Matrix:\n");
           for (i=0; i<N; i++)
           {
               printf("\n");
               for (j=0; j<N; j++)
                   printf("%6.2f   ", c[i][j]);
           }
       }
        */
      cout << "\n******************************************************\n";
      cout << " MPI Matrix size is " << N << endl;
      cout<< "MPI matrix multiplication runs in "
          << setprecision(8)
          << runtime << " seconds \n";
       
       printf ("Done.\n");
   }

     /**************************** worker task ************************************/
   if (taskid > MASTER)
   {
      mtype = FROM_MASTER;
      MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&(a[0][0]), rows*N, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&(b[0][0]), N*N, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

      for (k=0; k<N; k++)
         for (i=0; i<rows; i++)
         {
            c[i][k] = 0.0;
            for (j=0; j<N; j++)
               c[i][k] = c[i][k] + a[i][j] * b[j][k];
         }
      mtype = FROM_WORKER;
      MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&(c[0][0]), rows*N, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
   }
   MPI_Finalize();
}
