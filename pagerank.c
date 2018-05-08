#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "pr_graph.h"
#include <time.h>


/**
* @brief Compute the PageRank (PR) of a graph.
*
* @param graph The graph.
* @param damping Damping factor (or, 1-restart). 0.85 is typical.
* @param max_iterations The maximium number of iterations to perform.
*
* @return A vector of PR values.
*/
double * pagerank(
    pr_graph const * const graph,
    double const damping,
    int const max_iterations);


int main(int argc,char** argv)
{
	
	int max_iterations = 100;
	double tolerance = 1e-9;
	double damping = 0.85;
	
	int num_procs, myid;
	MPI_Init(&argc,&argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	
    char * ifname = argv[1];
    char * ofname = NULL;
	  if(argc > 2) {
		ofname = argv[2];
	  }
  
  pr_int nvtxs,nedges;
  pr_int* xadj, *nbrs;
  pr_int block_size;
  pr_int tot_vertex;
  double restart;
 
  if(myid == 0){
	  FILE * fin = fopen(ifname, "r");
	  fscanf(fin, "%lu", &nvtxs);
	  fscanf(fin, "%lu", &nedges);
	  fclose(fin);
  }
  
  MPI_Bcast(&nvtxs,1,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
  block_size = nvtxs/num_procs;
  tot_vertex = nvtxs;
  restart = (1-damping)/(double)nvtxs;

double timer1,timer2;
  if(myid == 0){
	  FILE * fin = fopen(ifname, "r");
		printf("master input and broadcast to each slaver\n");
		timer1 = clock();
	  /* read nvtxs and nedges */
	  fscanf(fin, "%lu", &nvtxs);
	  fscanf(fin, "%lu", &nedges);
	  fscanf(fin, "\n"); /* make sure we process the newline, too. */
	  
	  xadj = malloc((nvtxs+1)*sizeof(xadj));
	  nbrs = malloc(nedges*sizeof(nbrs));
	  
	  pr_int edge_ptr = 0;
	  
	  char * line = malloc(1024 * 1024);
	  size_t len = 0;

	  /* Read in graph one vertex at a time. */
	  for(pr_int v=0; v < nvtxs; ++v) {
		ssize_t read = getline(&line, &len, fin);

		  /* Store the beginning of the adjacency list. */
		  xadj[v] = edge_ptr;

		  /* Check for sinks -- these make pagerank more difficult. */
		  if(read == 1) {
			fprintf(stderr, "WARNING: vertex '%lu' is a sink vertex.\n", v+1);
			continue;
		  }

		  /* Foreach edge in line. */
		  char * ptr = strtok(line, " ");
		  while(ptr != NULL) {
			char * end = NULL;
			pr_int const e_id = strtoull(ptr, &end, 10);
			/* end of line */
			if(ptr == end) {
				break;
			}
			assert(e_id > 0 && e_id <= nvtxs);

			nbrs[edge_ptr++] = e_id - 1; /* 1 indexed */
			ptr = strtok(NULL, " ");
		  }
	  }
	  assert(edge_ptr == nedges);
	  xadj[nvtxs] = nedges;
	  free(line);
	  for(int i=1;i<num_procs;i++){
		  
		  pr_int Vnum = block_size;
		  if(i==num_procs-1){
			  Vnum = Vnum + nvtxs%block_size;
		  }
		  
		  int Begin = block_size * i;
		  int End = block_size * (i+1);
		  if(i==num_procs-1){
			  End = nvtxs;
		  }
		  
		  pr_int Enum = xadj[End] - xadj[Begin];
		  MPI_Send(&Vnum,1,MPI_UNSIGNED_LONG_LONG,i,0,MPI_COMM_WORLD);
		  MPI_Send(&Enum,1,MPI_UNSIGNED_LONG_LONG,i,0,MPI_COMM_WORLD);
	  }
  }
  else
  {
	  MPI_Recv(&nvtxs,1,MPI_UNSIGNED_LONG_LONG,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	  MPI_Recv(&nedges,1,MPI_UNSIGNED_LONG_LONG,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  if(myid != 0){
	  xadj = malloc((nvtxs+1)*sizeof(xadj));
	  nbrs = malloc(nedges*sizeof(nbrs));
  }
  MPI_Scatter(xadj,block_size,MPI_UNSIGNED_LONG_LONG,xadj,block_size,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
  if(myid==0){
	 	if(tot_vertex%block_size!=0) 
	  	MPI_Send(xadj+num_procs*block_size,tot_vertex%block_size,MPI_UNSIGNED_LONG_LONG,num_procs-1,0,MPI_COMM_WORLD);
	  
	  for(int i=1;i<num_procs;i++) {
		  int Begin = i * block_size;
		  int End = (i+1) * block_size;
		  if(i == num_procs-1){
			  End = nvtxs;
		  }
		  MPI_Send(nbrs+xadj[Begin],xadj[End]-xadj[Begin],MPI_UNSIGNED_LONG_LONG,i,0,MPI_COMM_WORLD);	  
	  }
	  /* rank 0 normalize */
	  nvtxs = block_size;
	  nedges = xadj[block_size];
  }
  else if(myid == num_procs-1){
		if(tot_vertex%block_size!=0)
	  	MPI_Recv(xadj+block_size,nvtxs-block_size,MPI_UNSIGNED_LONG_LONG,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	  MPI_Recv(nbrs,nedges,MPI_UNSIGNED_LONG_LONG,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
  else{
	  MPI_Recv(nbrs,nedges,MPI_UNSIGNED_LONG_LONG,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }

  xadj[nvtxs] = nedges+xadj[0]; 



	double* PR = malloc(nvtxs * sizeof(PR)); /* attention, *my* nvtxs = *your* nvtxs/num_procs */
	
	for(pr_int i=0;i<nvtxs;i++){
		PR[i] = 1./(double)tot_vertex;
	}
	
	double* inner_acc = malloc(nvtxs * sizeof(inner_acc));
	
	int* send_dis = malloc(num_procs*num_procs* sizeof(send_dis));
	int* recv_dis = malloc(num_procs*sizeof(recv_dis));

	int* ineedfrom = malloc(num_procs*sizeof(ineedfrom));
	int* isendto = malloc(num_procs*num_procs*sizeof(isendto));	
	
	int* send_idx = malloc(num_procs*sizeof(send_idx));

	memset(ineedfrom,0,num_procs*sizeof(ineedfrom));

	for(int i=0;i<num_procs;i++) isendto[i] = 1;

	for(int i=0;i<nedges;i++){
		int idx = nbrs[i]/block_size;
		if(idx==num_procs) --idx;
		isendto[idx] ++;
	}

	/* tell each processor how many data they are to receive*/
	for(int i=0;i<num_procs;i++)
		MPI_Scatter(isendto,1,MPI_INT,ineedfrom+i,1,MPI_INT,i,MPI_COMM_WORLD);



	int* buffer = malloc(num_procs*num_procs*sizeof(buffer));	
	MPI_Allgather(isendto,num_procs,MPI_INT,buffer,num_procs,MPI_INT,MPI_COMM_WORLD);
	memcpy(isendto,buffer,num_procs*num_procs*sizeof(buffer));
	/* mysterious bug, without the buffer, things go wrong only in num_procs = 2 setting*/


	for(int j=0;j<num_procs;j++){
		send_dis[j*num_procs] = 0;
		for(int i=1;i<num_procs;i++)			
			send_dis[j*num_procs+i] = send_dis[j*num_procs+i-1] + isendto[j*num_procs+i-1];
	}

	recv_dis[0] = 0;
	for(int i=1;i<num_procs;i++) {
			recv_dis[i] = recv_dis[i-1] + ineedfrom[i-1];
	}


	pr_int* recv_key = malloc((recv_dis[num_procs-1]+ineedfrom[num_procs-1])*sizeof(recv_key));
	double* recv_value = malloc((recv_dis[num_procs-1]+ineedfrom[num_procs-1])*sizeof(recv_value));
	
	pr_int* send_key = malloc((send_dis[(myid+1)*num_procs-1]+isendto[(myid+1)*num_procs-1])*sizeof(send_key));
	double* send_value = malloc((send_dis[(myid+1)*num_procs-1]+isendto[(myid+1)*num_procs-1])*sizeof(send_value));
	if(myid==0)
		printf("begin page rank, Input use time %.5fs\n",(clock()-timer1)/CLOCKS_PER_SEC);
	timer2 = clock();
	int iter;
	for(iter=0;iter<max_iterations;iter++){
		for(int i=0;i<num_procs;i++){
				send_key[send_dis[myid*num_procs+i]] = -1;
				send_idx[i] = 1;
		}
	
		for(int i=0;i<nvtxs;i++)
			inner_acc[i] = 0;

		for(pr_int v=0;v<nvtxs;v++){
			double num_links = (double)(xadj[v+1]-xadj[v]);
			double push_val = PR[v] / num_links;
			
			for(pr_int e=xadj[v];e<xadj[v+1];++e){
					int u = nbrs[e-xadj[0]];
					int idx = u/block_size;
					idx = (idx==num_procs?idx-1:idx);
					
					send_key[send_dis[myid*num_procs+idx]+send_idx[idx]] = u;
					send_value[send_dis[myid*num_procs+idx]+send_idx[idx]] = push_val;
					++send_idx[idx];
					
			}
		}

   for(int i=0;i<num_procs;i++){
			MPI_Scatterv(send_key,isendto+i*num_procs,send_dis+i*num_procs,MPI_UNSIGNED_LONG_LONG,recv_key+recv_dis[i],ineedfrom[i],MPI_UNSIGNED_LONG_LONG,i,MPI_COMM_WORLD);
			MPI_Scatterv(send_value,isendto+i*num_procs,send_dis+i*num_procs,MPI_DOUBLE,recv_value+recv_dis[i],ineedfrom[i],MPI_DOUBLE,i,MPI_COMM_WORLD);
		}
		for(int i=0;i<recv_dis[num_procs-1]+ineedfrom[num_procs-1];i++){
			if(recv_key[i] ==-1) continue;
			pr_int key = recv_key[i];
			double value = recv_value[i];
			inner_acc[key-myid*block_size]+=value;
		}

		double norm_change = 0;

		for(int i=0;i<nvtxs;i++){
			double old = PR[i];
			PR[i] = restart + (damping * inner_acc[i]);
			norm_change += (PR[i]-old)*(PR[i]-old);
		}
		double gather;
		MPI_Reduce(&norm_change,&gather,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		gather = sqrt(gather);
		MPI_Bcast(&gather,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		if(gather < tolerance){
				++iter;
				break;
		}
	}
	if(myid == 0)
		printf("Number of iterations: %d average time: %.3f\n",iter,1.0*(clock()-timer2)/CLOCKS_PER_SEC/iter);

  if(ofname) {
    FILE * fout;
		if(myid==0) fout = fopen(ofname, "w");
   	if(myid==0){
			for(int i=0;i<num_procs;i++) {
				if(i!=0){
					MPI_Recv(PR,block_size,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				}
				for(int j=0;j<block_size;j++){
					fprintf(fout,"%0.3e\n",PR[j]);
				}
				if(i!=0 && i==num_procs-1){
					MPI_Recv(PR,tot_vertex%block_size,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
					for(int j=0;j<tot_vertex%block_size;j++){
						fprintf(fout,"%0.3e\n",PR[j]);
					}
				}
			}
		}
		else{
			MPI_Send(PR,block_size,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
			if(myid == num_procs -1 ){
				MPI_Send(PR+block_size,tot_vertex%block_size,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
			}
		}

		if(myid==0)
    fclose(fout);
  }

	MPI_Finalize();
  return EXIT_SUCCESS;
}


