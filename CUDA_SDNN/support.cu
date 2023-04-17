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
  while (feof(fp) != true)
  {
      fgets(row, MAXCHAR, fp);
      printf("Reading row: %d", r_idx);
      token = strtok(row, ",");
      c_idx = 0;
      while(token != NULL)
      {
          // printf("Token: %s\n", token);
          printf("Reading col: %d", c_idx);
          W[r_idx][c_idx] = atof(token);
          token = strtok(NULL, ",");
          c_idx++;
      }
      r_idx++;
  }
  return W;
}







