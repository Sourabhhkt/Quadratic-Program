#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
void initVector(unsigned int **vec_h, unsigned int size, unsigned int num_bins);
void verify(float *A, float *B, float *C, int n);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
void print_1d_array(int _c, float* arr);
void print_1d_array_usint(int _c, unsigned int* arr);
void print_2d_array(int _r, int _c, float** arr);
float* g_function(int num_dim, float* z,  float* l,  float* h );
float* mat_mul_vec(int _r, int _c, float** _A,float* _vec);
float* vec_add_vec(int _r, float* _vec1,float* _vec2);
float* scale_vec(int _r, float scaler, float* _vec);
float* convert_2d_mat_to_1d_arr(int row_num, int col_num,float** ME_T);
float vec_l1_norm(int col_num, float* gradient);
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
