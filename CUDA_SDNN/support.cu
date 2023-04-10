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
      printf("%.1f ",arr[j]);
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
          printf("%.3f ", arr[i][j]);
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
  float[num_dim] ans; 
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
