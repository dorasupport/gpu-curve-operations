#ifndef FOO_CUH
#define FOO_CUH

#include <stdio.h>
typedef unsigned char uint8_t;

extern "C" {
int cudaDo(int argc, const char* argv[]);
}
int do_calc_np_sigma(int n, std::vector<uint8_t *> scaler, std::vector<uint8_t *> x1, std::vector<uint8_t *> y1, std::vector<uint8_t *> z1, uint8_t *x3, uint8_t *y3, uint8_t *z3);

#endif




