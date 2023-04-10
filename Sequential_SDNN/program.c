#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <support.h>


// Solve QP Problem with SDNN sequentially
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

