#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// function declaration
float random_from_max(float _m);
void print_2d_array(int _r, int _c, float** arr);
float** random_mat(int _r, int _c, float _m);
void print_mat_mul( int _r, int _c, float** _A,float** _B);
void print_1d_array(int _c, float* arr);

int main( int argc, char *argv[] )  {
    srand ( clock()+ (time(0)*time(0)) );
    // srand ( time(NULL) );
    int val_input = (argc-1)%3;
    if(val_input==0 && argc > 1){
        // int triple_count = (argc-1)/3;
        // printf("No. of input triples %d\n\n",triple_count);
        // printf("Rand no from rand(): %d",rand());
        int i,r,c;
        float m;
        for (i = 1; i < argc; i = i+3){
            r = atoi(argv[i]);
            c = atoi(argv[i+1]);
            m = atof(argv[i+2]);
            // printf("Triple %d-th = [%d ,%d, %.3f]\n",(i+2)/3,r,c,m);
            // initialize array: Mat A, B
            float** A;
            float** B;
            // Rand for A
            A = random_mat(r,c,m);
            B = random_mat(c,r,m);

            // printf("Mat A:\n");
            // print_2d_array(r,c,A);
            // printf("\nMat B:\n");
            // print_2d_array(c,r,B);
            // printf("\n");

            print_mat_mul(r,c,A,B);

            // printf("Triple %d-th = [%s ,%s, %s]\n",i,argv[i],argv[i+1],argv[i+2]);
        }
        return 0;
    } else{
        printf("Invalid number of arguments.\n");
        return 0;
    }
}

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