#ifndef FOO_CUH
#define FOO_CUH

#include <stdio.h>
typedef unsigned char uint8_t;

extern "C" {
int cudaDo(int argc, const char* argv[]);
}
int do_calc_np_sigma(int n, uint8_t* scaler, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t *x3, uint8_t *y3, uint8_t *z3);

#endif




