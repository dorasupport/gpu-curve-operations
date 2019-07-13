#ifndef FOO_CUH
#define FOO_CUH

#include <stdio.h>
typedef unsigned char uint8_t;

extern "C" {
int cudaDo(int argc, const char* argv[]);
}
int mnt4_g1_pq_plus(int n, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t* x2, uint8_t* y2, uint8_t* z2, uint8_t *x3, uint8_t *y3, uint8_t *z3);
int mnt4_g1_do_calc_np_sigma(int n, uint8_t* scalar, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t *x3, uint8_t *y3, uint8_t *z3);
int mnt6_g1_do_calc_np_sigma(int n, uint8_t* scalar, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t *x3, uint8_t *y3, uint8_t *z3);
int mnt4_g2_do_calc_np_sigma(int n, uint8_t * scalar, uint8_t* x10, uint8_t* x11, uint8_t* y10, uint8_t* y11, uint8_t* z10, uint8_t* z11, uint8_t *x30, uint8_t *x31, uint8_t *y30, uint8_t *y31, uint8_t *z30, uint8_t *z31);
int mnt6_g2_do_calc_np_sigma(int n, uint8_t * scalar, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t *x3, uint8_t *y3, uint8_t *z3);
int mnt4_g2_mul(int n, uint8_t *x1, uint8_t *x2, uint8_t *y1, uint8_t *y2, uint8_t *o1, uint8_t *o2);
int mnt6_g2_mul(int n, uint8_t *x1, uint8_t *x2, uint8_t *x3, uint8_t *y1, uint8_t *y2, uint8_t *y3, uint8_t *o1, uint8_t *o2, uint8_t *o3);
#endif
