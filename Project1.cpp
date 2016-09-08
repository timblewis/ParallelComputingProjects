#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <string>

#ifndef OPT
#include "mpi.h"
#else
#include <optmpi.h>
#endif

#define PI 3.14159265

using namespace std;

int main(int argc, char **argv){

	int rank;          /* rank of process */
	int numprocs;      /* size of COMM_WORLD */
	int tag = 10;        /* expected tag*/
	//get number of rows from console argument
	int n;
	if(argc < 2){
		n = 1000;
	} else {
		istringstream ss(argv[1]);
		if (!(ss >> n)){
			n = 1000;
		}
	}
	
	//initialize MPI stuff
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int size = (rank+1)*n/numprocs - rank*n/numprocs;
	
	//initialize A
	double** A = new double*[size];
	for(int k = 0; k < size; k++){
		A[k] = new double[n];
	}

	for(int k = 0; k < size; k++){
		for(int l = 0; l < n; l++){
			A[k][l] = 0.5;
		}
	}
	if(rank == 0){
		for(int k = 0; k < n; k++){
			A[0][k] = 0;
		}
	}
	if(rank == numprocs-1){
		for(int k = 0; k < n; k++){
			A[size-1][k] = 5*sin(PI*k*k/(n*n));
		}
	}
	
	//Use MPI_Barrier and MPI_WTime to start the clock
	MPI_Barrier(MPI_COMM_WORLD);
	double startTime = MPI_Wtime();
	
	//iterations
	for(int iteration = 0; iteration < 500; iteration++){
		//send data
		MPI_Request rightSendRequest, leftSendRequest;
		if(rank > 0){
			//send to left
			MPI_Isend(A[0],          /*send data */
				n,          /*number to send */
				MPI_DOUBLE,          /*type to send */
				rank-1,          /*rank to send to */
				tag,          /*tag for message */
				MPI_COMM_WORLD, /*communicator to use */
				&leftSendRequest);        
		}
		if(rank < numprocs-1){
			//send to right
			MPI_Isend(A[size-1],          /*send data */
				n,          /*number to send */
				MPI_DOUBLE,          /*type to send */
				rank+1,          /*rank to send to */
				tag,          /*tag for message */
				MPI_COMM_WORLD, /*communicator to use */
				&rightSendRequest); 
		}
		
		//non-blocking receive data
		MPI_Request leftReceiveRequest;
		double* leftData = new double[n];
		MPI_Request rightReceiveRequest;
		double* rightData = new double[n];
		if(rank > 0){
			//receive from left
			MPI_Irecv(&leftData[0],    /*buffer for message */
			n,    /*MAX count to recv */
			MPI_DOUBLE,    /*type to recv */
			rank-1,    /*recv from 0 only */
			tag,    /*tag of message */
			MPI_COMM_WORLD,    /*communicator to use */
			&leftReceiveRequest);   /*status object */
		}
		if(rank < numprocs-1){
			//receive from right
			MPI_Irecv(&rightData[0],    /*buffer for message */
			n,    /*MAX count to recv */
			MPI_DOUBLE,    /*type to recv */
			rank+1,    /*recv from 0 only */
			tag,    /*tag of message */
			MPI_COMM_WORLD,    /*communicator to use */
			&rightReceiveRequest);   /*status object */
		}
		
		//calculate inner data
		double** B = new double*[size]; //where we are temporarily storing the data
		for(int i = 0; i < size; i++){
			B[i] = new double[n];
		}

		for(int i = 1; i < size-1;i++){
			for(int j = 0; j < n; j++){
				B[i][j] = (A[i-1][(j-1+n)%n] + A[i][(j-1+n)%n] + A[i+1][(j-1+n)%n]
						+ A[i-1][j] + A[i][j] + A[i+1][j]
						+ A[i-1][(j+1)%n] + A[i][(j+1)%n] + A[i+1][(j+1)%n])/9; 
			}
		}
		
		//wait for receive
		MPI_Status leftStatus;
		MPI_Status rightStatus;
		if(rank > 0){
			//wait for left
			MPI_Wait(&leftReceiveRequest, &leftStatus);
		}
		if(rank < numprocs-1){
			//wait for right
			MPI_Wait(&rightReceiveRequest, &rightStatus);
		}
		
		//calculate left and right most column
		if(rank > 0){
			//wait to make sure left has been sent
			MPI_Wait(&leftSendRequest, &leftStatus);
			//calculate left otherwise it stays the same
			for(int i=0; i < n; i++){
				B[0][i] = (leftData[(i-1+n)%n] + A[0][(i-1+n)%n] + A[1][(i-1+n)%n]
						+ leftData[i] + A[0][i] + A[1][i]
						+ leftData[(i+1)%n] + A[0][(i+1)%n] + A[1][(i+1)%n])/9;
			}
		} else {
			for(int i = 0; i < n; i++){
				B[0][i] = A[0][i];
			}
		}
		if(rank < numprocs-1){
			//wait to make sure right has been sent
			MPI_Wait(&rightSendRequest, &rightStatus);
			//calculate right otherwise it stays the same
			for(int i = 0; i < n; i++){
				B[size-1][i] = (A[size-2][(i-1+n)%n] + A[size-1][(i-1+n)%n] + rightData[(i-1+n)%n]
						+ A[size-2][i] + A[size-1][i] + rightData[i]
						+ A[size-2][(i+1)%n] + A[size-1][(i+1)%n] + rightData[(i+1)%n])/9; 
			}
		} else {
			for(int i = 0; i < n; i++){
				B[size-1][i] = A[size-1][i];
			}
		}
		
		//De-Allocate old A
		for(int i = 0; i < size; i++){
			delete [] A[i];
		}
		delete [] A;
		delete [] leftData;
		delete [] rightData;
		
		//set new data
		A = B;
	}
	
	//calculate local verification value
	double localVerificationSum = 0.0;
	for(int i = 0; i < size; i++){
		localVerificationSum += A[i][i + rank*n/numprocs];
	}
	
	//calculate global verification value
	double globalVerificationSum = 0.0;
	MPI_Reduce(&localVerificationSum, //where from
		&globalVerificationSum, //where to
		1, //count
		MPI_DOUBLE, //type
		MPI_SUM, //operator
		0, //root
		MPI_COMM_WORLD); //communicator
	
	//use MPI_Barrier and MPI_WTime to stop the clock
	MPI_Barrier(MPI_COMM_WORLD);
	double endTime = MPI_Wtime();
	double totalTime = endTime - startTime;
	
	//output verification sum and clock time
	if(rank == 0){
		cout << "For n=" << n << " and p=" << numprocs << ":" << endl 
			 << "The verification sum is " << globalVerificationSum << endl 
			 << "This was computed in " << totalTime << " seconds" << endl;
	}

	MPI_Finalize();

	return 0;
}
