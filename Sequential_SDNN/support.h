#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif


// function declaration
void initVector(unsigned int **vec_h, unsigned int size, unsigned int num_bins);
void verify(float *A, float *B, float *C, int n);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

float random_from_max(float _m);
void print_2d_array(int _r, int _c, float** arr);
float** random_mat(int _r, int _c, float _m);
void print_mat_mul( int _r, int _c, float** _A,float** _B);
void print_1d_array(int _c, float* arr);


void print_mat_mul( int _r, int _c, float** _A,float** _B){
    int _i,_j,_k;
    for (_i = 0; _i < _r; _i++){
        float ans_arr[_r];
        for (_k = 0; _k < _r; _k++){
            float _s = 0;
            for (_j = 0; _j < _c; _j++){
                _s = _s + _A[_i][_j]*_B[_j][_k];
                // printf("%f ",_A[_i][_j]*_B[_j][_k]);
                // printf("A[%d][%d] x B[%d][%d] = %.3f x %.3f = %f \n",_i,_j,_j,_k, _A[_i][_j],_B[_j][_k],_A[_i][_j]*_B[_j][_k]);
            }
            ans_arr[_k] = _s;
        }
        print_1d_array(_r, ans_arr);
    }
    // printf("\n");

    for (_i = 0; _i < _r; _i++){
        for (_k = 0; _k < _r; _k++){
            for (_j = 0; _j < _c; _j++){
                // printf("A[%d][%d] x B[%d][%d] = %.3f x %.3f = %f \n",_i,_j,_j,_k, _A[_i][_j],_B[_j][_k],_A[_i][_j]*_B[_j][_k]);
            }
        }
    }
}



float random_from_max(float _m){
    float _rand = (((float)rand()/(float)(RAND_MAX)) * _m );
    // float _rand = (((float)rand(clock())/(float)(RAND_MAX)) * _m );
    _rand = ceil(_rand * 1000.0 ) / 1000.0;
    return _rand;
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

void print_1d_array(int _c, float* arr){
    int j;
    for (j = 0; j < _c; j++){
        printf("%f ",arr[j]);
    }
    printf("\n");
}

float** random_mat(int _r, int _c, float _m){\
    int _i, _j;
    float** arr;
    arr = (float**)malloc(sizeof(float*)*_r);
    for (_i = 0; _i < _r; _i++){
        *(arr+_i) = (float*)malloc(sizeof(float)*_c);
    }
    for (_i = 0; _i < _r; _i++){
        for (_j = 0; _j < _c; _j++){
                arr[_i][_j] = random_from_max(_m);
                // printf("%f ", arr[_i][_j]);
        }
    }
    return arr;
}




#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
    do {\
      fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__); \
      exit(-1);								\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
