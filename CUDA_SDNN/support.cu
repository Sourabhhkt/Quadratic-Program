#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#include "support.h"

void verify(float *A, float *B, float *C, int n) {

  const float relativeTolerance = 1e-6;

  for(int i = 0; i < n; i++) {
    float sum = A[i] + B[i];
    float relativeError = (sum - C[i])/sum;
    if (relativeError > relativeTolerance
      || relativeError < -relativeTolerance) {
      printf("TEST FAILED\n\n");
      exit(0);
    }
  }
  printf("TEST PASSED\n\n");

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}
void print_1d_array(int _c, float* arr){
  int j;
  for (j = 0; j < _c; j++){
      printf("%.5f ",arr[j]);
  }
  printf("\n");
}

void print_1d_array_usint(int _c, unsigned int* arr){
  int j;
  for (j = 0; j < _c; j++){
      printf("%d ",arr[j]);
  }
  printf("\n");
}

void print_2d_array(int _r, int _c, float** arr){
  int i, j;
  for (i = 0; i < _r; i++){
      for (j = 0; j < _c; j++){
          printf("%.5f ", arr[i][j]);
      }
      printf("\n");
  }
}
// def g_func(z, l, h):
//     g = []
//     for i in range(len(z)):
//         if ((l[i] == -np.inf) and (h[i]==np.inf)):
//             g.append(z[i])
//         elif (l[i] == -np.inf):
//             if (z[i]<= h[i]): g.append(z[i])
//             else: g.append(h[i])
//         elif (h[i]==np.inf):
//             if (z[i]>= l[i]): g.append(z[i])
//             else: g.append(l[i])
//         else:
//             if (z[i]< l[i]): g.append(l[i])
//             elif ((l[i]<=z[i])and(z[i]<=h[i])): g.append(z[i])
//             else: g.append(h[i])
//     return np.array(g)

float* g_function(int num_dim, float* z,  float* l,  float* h )
{
  float* ans = (float*) malloc( sizeof(float)*num_dim );
  int i;
  for (i = 0; i < num_dim; i++)
  {
    if ((l[i] == FLT_MIN) and (h[i]== FLT_MAX)) {
      ans[i] = z[i];
    }else if (l[i] == FLT_MIN)
    {
      if (z[i] <= h[i])
      {
        ans[i] = z[i];
      }else{
        ans[i] = h[i];
      }
    }else if (h[i] == FLT_MAX){
      if (z[i] >= l[i])
      {
        ans[i] = z[i];
      }else{
        ans[i] = l[i];
      }
    }else{
      if (z[i] < l[i])
      {
        ans[i] = l[i];
      }else if ((l[i]<=z[i]) and (z[i]<=h[i])) {
        ans[i] = z[i];
      }else{
        ans[i] = h[i];
      }
    }
  }
  return ans;
}

float* mat_mul_vec(int _r, int _c, float** _A,float* _vec)
{
  int _i,_j;
  float* ans_arr = (float*)malloc(sizeof(float)*_r);;
  for (_i = 0; _i < _r; _i++)
  {
    float _s = 0;
    for (_j = 0; _j < _c; _j++){
        _s = _s + _A[_i][_j]*_vec[_j];
        // printf("%f ",_A[_i][_j]*_B[_j][_k]);
        // printf("A[%d][%d] x B[%d][%d] = %.3f x %.3f = %f \n",_i,_j,_j,_k, _A[_i][_j],_B[_j][_k],_A[_i][_j]*_B[_j][_k]);
    }
    ans_arr[_i] = _s;   
  }
  // print_1d_array(_r, ans_arr);
  return ans_arr;
}

float* vec_add_vec(int _r, float* _vec1,float* _vec2)
{
  int _j;
  float* ans_vec = (float*)malloc(sizeof(float)*_r);
  for (_j = 0; _j < _r; _j++){
    ans_vec[_j] = _vec1[_j] + _vec2[_j];
  }
  // print_1d_array(_r, ans_vec);
  return ans_vec;
}

float* scale_vec(int _r, float scaler, float* _vec)
{
  int _j;
  float* ans_vec = (float*)malloc(sizeof(float)*_r);
  for (_j = 0; _j < _r; _j++){
    ans_vec[_j] = scaler*_vec[_j];
  }
  // print_1d_array(_r, ans_vec);
  return ans_vec;
}

float* convert_2d_mat_to_1d_arr(int row_num, int col_num,float** ME_T){
  float* mat_1d = (float*) malloc( sizeof(float)*row_num*col_num);
  for (int i = 0; i < row_num; i++)
  {
    for (int j = 0; j < col_num; j++)
    {
      mat_1d[i*col_num + j] = ME_T[i][j];
    }
  }
  return mat_1d;
}

float vec_l1_norm(int col_num, float* gradient){
  float ans;
  for (int j = 0; j < col_num; j++)
  {
    ans += abs(gradient[j]);
  }
  return ans;
}

#define MAXCHAR 100000
void read_csv_file(char* filename){ 

  FILE *fp;
  char row[MAXCHAR];
  char *token;

  fp = fopen(filename,"r");


  while (feof(fp) != true)
  {
      fgets(row, MAXCHAR, fp);
      printf("Row: %s", row);

      token = strtok(row, ",");

      while(token != NULL)
      {
          printf("Token: %s\n", token);
          token = strtok(NULL, ",");
      }

  }
}

#define STR_SIZE 10000
#define DELIM " ,\n"
float** read_W(char* inst_path,char* inst_name, int row_num, int col_num){ 

  float** W = (float**) malloc( sizeof(float*)*row_num );
  for (int i = 0; i < row_num; i++){
      *(W+i) = (float*)malloc(sizeof(float)*row_num);
  }
  FILE *fp;
  char row[MAXCHAR];
  char *token;

  char filepath[STR_SIZE] = {0};
  snprintf(filepath, sizeof(filepath), "%s%s%s", inst_path, inst_name, "_W1.csv");
  char* pointer_to_path = filepath;

  fp = fopen(pointer_to_path,"r");
  printf("Reading file... %s \n", filepath);

  int r_idx = 0; int c_idx = 0;
  int read_r_size;int read_c_size;
  while (feof(fp) != true)
  {
      fgets(row, MAXCHAR, fp);
      printf("Reading row: %d \n", r_idx);
      token = strtok(row, DELIM);
      c_idx = 0;
      while(token != NULL)
      {
          printf("   Reading col: %d \n", c_idx);
          if ((c_idx < row_num) and (r_idx < row_num)) {
            W[r_idx][c_idx] = atof(token);
            read_r_size = r_idx; 
            read_c_size = c_idx;
          }
          token = strtok(NULL, DELIM);
          c_idx++;
      }
      r_idx++;
  }
  printf("Finished reading W with size r:%d x c:%d \n", read_r_size+1, read_c_size+1);
  return W;
}

float** read_E(char* inst_path,char* inst_name, int row_num, int col_num){ 

  float** E = (float**) malloc( sizeof(float*)*row_num );
  for (int i = 0; i < row_num; i++){
      *(E+i) = (float*)malloc(sizeof(float)*col_num);
  }
  FILE *fp;
  char row[MAXCHAR];
  char *token;

  char filepath[STR_SIZE] = {0};
  snprintf(filepath, sizeof(filepath), "%s%s%s", inst_path, inst_name, "_E.csv");
  char* pointer_to_path = filepath;

  fp = fopen(pointer_to_path,"r");
  printf("Reading file... %s \n", filepath);

  int r_idx = 0; int c_idx = 0;
  int read_r_size;int read_c_size;
  while (feof(fp) != true)
  {
      fgets(row, MAXCHAR, fp);
      // printf("Reading row: %d \n", r_idx);
      token = strtok(row, DELIM);
      c_idx = 0;
      while(token != NULL)
      {
          // printf("   Reading col: %d \n", c_idx);
          if ((c_idx < col_num) and (r_idx < row_num)) {
            E[r_idx][c_idx] = atof(token);
            read_r_size = r_idx; 
            read_c_size = c_idx;
          }
          token = strtok(NULL, DELIM);
          c_idx++;
      }
      r_idx++;
  }
  printf("Finished reading E with size r:%d x c:%d \n", read_r_size+1, read_c_size+1);
  return E;
}

float* read_C(char* inst_path,char* inst_name, int row_num){ 

  float* C = (float*) malloc( sizeof(float)*row_num );
  
  FILE *fp;
  char row[MAXCHAR];
  char *token;

  char filepath[STR_SIZE] = {0};
  snprintf(filepath, sizeof(filepath), "%s%s%s", inst_path, inst_name, "_C.csv");
  char* pointer_to_path = filepath;

  fp = fopen(pointer_to_path,"r");
  printf("Reading file... %s \n", filepath);

  int r_idx = 0;int c_idx = 0;
  int read_r_size;int read_c_size;
  while (feof(fp) != true)
  {
      fgets(row, MAXCHAR, fp);
      // printf("Reading row: %d \n", r_idx);
      token = strtok(row, DELIM);
      c_idx = 0;
      while(token != NULL)
      {
          // printf("   Reading col: %d \n", c_idx);
          if ((c_idx < 1) and (r_idx < row_num)) {
            C[r_idx] = atof(token);
            read_r_size = r_idx; 
            read_c_size = c_idx;
          }
          token = strtok(NULL, DELIM);
          c_idx++;
      }
      r_idx++;
  }
  printf("Finished reading C with size r:%d x c:%d \n", read_r_size+1, read_c_size+1);
  return C;
}

float* read_l(char* inst_path,char* inst_name, int row_num){ 

  float* l = (float*) malloc( sizeof(float)*row_num );
  
  FILE *fp;
  char row[MAXCHAR];
  char *token;

  char filepath[STR_SIZE] = {0};
  snprintf(filepath, sizeof(filepath), "%s%s%s", inst_path, inst_name, "_l.csv");
  char* pointer_to_path = filepath;

  fp = fopen(pointer_to_path,"r");
  printf("Reading file... %s \n", filepath);
  char infstr[] = "inf";
  char minusinfstr[] = "-inf";

  int r_idx = 0;int c_idx = 0;
  int read_r_size;int read_c_size;
  while (feof(fp) != true)
  {
      fgets(row, MAXCHAR, fp);
      // printf("Reading row: %d \n", r_idx);
      token = strtok(row, DELIM);
      c_idx = 0;
      while(token != NULL)
      {
          // printf("   Reading col: %d \n", c_idx);
          if ((c_idx < 1) and (r_idx < row_num)) {
            if (strcmp(infstr,token)==0){
              l[r_idx] = FLT_MAX;
            } else if(strcmp(minusinfstr,token)==0) {
              l[r_idx] = FLT_MIN;
            }else {
              l[r_idx] = atof(token);
            }
            read_r_size = r_idx; 
            read_c_size = c_idx;
          }
          token = strtok(NULL, DELIM);
          c_idx++;
      }
      r_idx++;
  }
  printf("Finished reading l with size r:%d x c:%d \n", read_r_size+1, read_c_size+1);
  return l;
}

float* read_h(char* inst_path,char* inst_name, int row_num){ 

  float* h = (float*) malloc( sizeof(float)*row_num );
  
  FILE *fp;
  char row[MAXCHAR];
  char *token;

  char filepath[STR_SIZE] = {0};
  snprintf(filepath, sizeof(filepath), "%s%s%s", inst_path, inst_name, "_h.csv");
  char* pointer_to_path = filepath;

  fp = fopen(pointer_to_path,"r");
  printf("Reading file... %s \n", filepath);
  char infstr[] = "inf";
  char minusinfstr[] = "-inf";

  int r_idx = 0;int c_idx = 0;
  int read_r_size;int read_c_size;
  while (feof(fp) != true)
  {
      fgets(row, MAXCHAR, fp);
      // printf("Reading row: %d \n", r_idx);
      token = strtok(row, DELIM);
      c_idx = 0;
      while(token != NULL)
      {
          // printf("   Reading col: %d \n", c_idx);
          if ((c_idx < 1) and (r_idx < row_num)) {
            if (strcmp(infstr,token)==0){
              h[r_idx] = FLT_MAX;
            } else if(strcmp(minusinfstr,token)==0) {
              h[r_idx] = FLT_MIN;
            }else {
              h[r_idx] = atof(token);
            }
            read_r_size = r_idx; 
            read_c_size = c_idx;
          }
          token = strtok(NULL, DELIM);
          c_idx++;
      }
      r_idx++;
  }
  printf("Finished reading h with size r:%d x c:%d \n", read_r_size+1, read_c_size+1);
  return h;
}
float** inverse_mat(float** A, int mat_row_and_col){
  float **I,temp;
  int i,j,k;
  int matsize = mat_row_and_col;
  // printf("Enter the size of the matrix(i.e. value of 'n' as size is nXn):");
  // scanf("%d",&matsize);
  // A=(float **)malloc(matsize*sizeof(float *));            //allocate memory dynamically for matrix A(matsize X matsize)
  // for(i=0;i<matsize;i++)
  //     A[i]=(float *)malloc(matsize*sizeof(float));

  I=(float **)malloc(matsize*sizeof(float *));            //memory allocation for indentity matrix I(matsize X matsize)
  for(i=0;i<matsize;i++)
      I[i]=(float *)malloc(matsize*sizeof(float));

  // printf("Enter the matrix: ");                           // ask the user for matrix A
  // for(i=0;i<matsize;i++)
  //     for(j=0;j<matsize;j++)
  //         scanf("%f",&A[i][j]);

  for(i=0;i<matsize;i++)                                  //automatically initialize the unit matrix, e.g.
      for(j=0;j<matsize;j++)                              //  -       -
          if(i==j)                                        // | 1  0  0 |
              I[i][j]=1;                                  // | 0  1  0 |
          else                                            // | 0  0  1 |
              I[i][j]=0;                                  //  -       -
  /*---------------LoGiC starts here------------------*/      //procedure // to make the matrix A to unit matrix

  for(k=0;k<matsize;k++)                                  //by some row operations,and the same row operations of
  {                                                       //Unit mat. I gives the inverse of matrix A
      temp=A[k][k];     //'temp'  
      printf("temp %f \n", temp);
      // stores the A[k][k] value so that A[k][k]  will not change
      for(j=0;j<matsize;j++)      //during the operation //A[i] //[j]/=A[k][k]  when i=j=k
      {
          A[k][j]/=temp;                                  //it performs // the following row operations to make A to unit matrix
          I[k][j]/=temp;                                  //R0=R0/A[0][0],similarly for I also R0=R0/A[0][0]
      }                                                   //R1=R1-R0*A[1][0] similarly for I
      for(i=0;i<matsize;i++)                              //R2=R2-R0*A[2][0]      ,,
      {
          temp=A[i][k];                       //R1=R1/A[1][1]
          for(j=0;j<matsize;j++)             //R0=R0-R1*A[0][1]
          {                                   //R2=R2-R1*A[2][1]
              if(i==k)
                  break;                      //R2=R2/A[2][2]
              A[i][j]-=A[k][j]*temp;          //R0=R0-R2*A[0][2]
              I[i][j]-=I[k][j]*temp;          //R1=R1-R2*A[1][2]
          }
      }
  }
  /*---------------LoGiC ends here--------------------*/

  printf("Finished calculating the inverse...");
  return I;
}


float** mat_mul_mat( int _r, int _c, float** _A,float** _B){
  int _i,_j,_k;
  float** result = (float**) malloc( sizeof(float*)*_r );
  for (int i = 0; i < _r; i++){
        *(result+i) = (float*)malloc(sizeof(float)*_c);
  }

  for (_i = 0; _i < _r; _i++){
      // float ans_arr[_r];
      for (_k = 0; _k < _r; _k++){
          float _s = 0;
          for (_j = 0; _j < _c; _j++){
              _s = _s + _A[_i][_j]*_B[_j][_k];
              result[_i][_k] = _s;
              // printf("%f ",_A[_i][_j]*_B[_j][_k]);
              // printf("A[%d][%d] x B[%d][%d] = %.3f x %.3f = %f \n",_i,_j,_j,_k, _A[_i][_j],_B[_j][_k],_A[_i][_j]*_B[_j][_k]);
          }
          
      }
      // print_1d_array(_r, ans_arr);
  }
  // printf("\n");

  return result;
}

float** transpose_mat(int _r, int _c, float** _A){
  float** result = (float**) malloc( sizeof(float*)*_c );
  for (int i = 0; i < _c; i++){
        *(result+i) = (float*)malloc(sizeof(float)*_r);
  }

  for (int _i = 0; _i < _r; _i++){
    for (int _j = 0; _j < _c; _j++){
      result[_j][_i] = _A[_i][_j];
    }
  }
  return result;
}



