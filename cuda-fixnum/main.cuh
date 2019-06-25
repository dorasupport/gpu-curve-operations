#ifndef FOO_CUH
#define FOO_CUH

#include <stdio.h>
typedef unsigned char uint8_t;

extern "C" {
int cudaDo(int argc, const char* argv[]);
}
int do_calc_np(int n, uint8_t* scaler, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t *x3, uint8_t *y3, uint8_t *z3);
int do_calc_np_sigma(int n, uint8_t* scaler, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t *x3, uint8_t *y3, uint8_t *z3);
int do_calc_np_sigma_mnt4_g2(int n, uint8_t * scaler, uint8_t* x10, uint8_t* x11, uint8_t* y10, uint8_t* y11, uint8_t* z10, uint8_t* z11, uint8_t *x30, uint8_t *x31, uint8_t *y30, uint8_t *y31, uint8_t *z30, uint8_t *z31);
#endif




