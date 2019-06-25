#include <cstdio>
#include <cstring>
#include <cassert>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"
#include "mnt4_g1.cu"
#include "mnt4_g2.cu"
#include "mnt6_g1.cu"
#include "mnt6_g2.cu"

using namespace std;
using namespace cuFIXNUM;
using namespace MNT_G;
int BLOCK_NUM = 4096;
#define MNT_SIZE (96)
#define PARALLEL_SIGMA

// mnt4_q
const uint8_t mnt4_modulus[MNT_SIZE] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0};

// mnt6_q
uint8_t mnt6_modulus[MNT_SIZE] = {1,0,0,64,226,118,7,217,79,58,161,15,23,153,160,78,151,87,0,63,188,129,195,214,164,58,153,52,118,249,223,185,54,38,33,41,148,202,235,62,155,169,89,200,40,92,108,178,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0};

// mnt4 a, for calc 2p, not use now
uint8_t mnt4_a[MNT_SIZE] = {0x84,0xde,0xb8,0xb3,0x57,0xd9,0x51,0x31,0xd,0x8d,0x6,0xb4,0x8c,0x63,0x9a,0x23,0x5d,0xae,0x28,0x9a,0x41,0xc9,0x87,0x2f,0x3,0x6c,0x11,0x8f,0x33,0x30,0xb1,0xf2,0xde,0x2e,0x11,0x42,0x28,0x39,0x4d,0xda,0xd1,0x3a,0x06,0x9e,0x15,0x9b,0x1e,0x3c,0xb2,0xa,0x67,0x26,0x6e,0x77,0x18,0x64,0xc4,0x14,0xe0,0xa5,0x05,0x86,0x16,0xb3,0x42,0x4c,0x19,0xfb,0x97,0x93,0xe9,0x80,0x18,0xd1,0xcb,0x70,0xb6,0xfd,0x48,0x1f,0x2a,0x43,0xf3,0x3f,0x66,0xbf,0x8a,0x2a,0x85,0xc4,0x91,0x3d,0x8f,0xf6};

template< typename fixnum >
__device__ void dump(fixnum n, int size) {
#if 0
	for (int i = 0; i < size; i++) {
		printf("DUMP [%d] %x\n", i, fixnum::get(n, i));
	}
#endif
	printf("dump [%d]=\%x\n", threadIdx.x, fixnum::get(n, threadIdx.x));
}

template< typename fixnum >
struct mnt4g1_pq_plus {
    __device__ void operator()(fixnum mod, fixnum x1, fixnum y1, fixnum z1, fixnum x2, fixnum y2, fixnum z2, fixnum &x3, fixnum &y3, fixnum &z3) {
        typedef mnt4_g1<fixnum> mnt4g1;
        mnt4g1::pq_plus(mod, x1, y1, z1, x2, y2, z2, x3, y3, z3);
  }
};

template< typename fixnum >
struct mnt4g2_pq_plus {
    __device__ void operator()(fixnum mod, fixnum x10, fixnum x11, fixnum y10, fixnum y11, fixnum z10, fixnum z11, fixnum x20, fixnum x21, fixnum y20, fixnum y21, fixnum z20, fixnum z21, fixnum &x30, fixnum &x31, fixnum &y30, fixnum &y31, fixnum &z30, fixnum &z31) {
        typedef mnt4_g2<fixnum> mnt4g2;
        mnt4g2::pq_plus(mod, x10, x11, y10, y11, z10, z11, x20, x21, y20, y21, z20, z21, x30, x31, y30, y31, z30, z31);
  }
};

template< typename fixnum >
struct mnt6g1_pq_plus {
    __device__ void operator()(fixnum mod, fixnum x1, fixnum y1, fixnum z1, fixnum x2, fixnum y2, fixnum z2, fixnum &x3, fixnum &y3, fixnum &z3) {
        typedef mnt6_g1<fixnum> mnt6g1;
        mnt6g1::pq_plus(mod, x1, y1, z1, x2, y2, z2, x3, y3, z3);
  }
};

template< typename fixnum >
struct mnt6g2_pq_plus {
    __device__ void operator()(fixnum mod, fixnum x1, fixnum y1, fixnum z1, fixnum x2, fixnum y2, fixnum z2, fixnum &x3, fixnum &y3, fixnum &z3) {
        typedef mnt6_g2<fixnum> mnt6g2;
        mnt6g2::pq_plus(mod, x1, y1, z1, x2, y2, z2, x3, y3, z3);
  }
};

template< typename fixnum >
struct mnt4g1_calc_np {
    __device__ void operator()(fixnum mod, fixnum w, fixnum x1, fixnum y1, fixnum z1, fixnum &x3, fixnum &y3, fixnum &z3) {
    typedef modnum_monty_cios<fixnum> modnum;
    typedef mnt4_g1<fixnum> mnt4g1;
    modnum m(mod);
    fixnum rx, ry, rz;
    fixnum tempw = w;
    int i = 24*32 - 1;
    bool found_one = false;
    int count = 0;
#if 0
    if (threadIdx.x > 23) {
        dump(w, 24);
        dump(x1, 24);
        dump(y1, 24);
        dump(z1, 24);
    }
#endif
    //while(fixnum::cmp(tempw, fixnum::zero()) && i >= 0) {
    while(i >= 0) {
        size_t value = fixnum::get(tempw, i/32);
#if 0
        if (threadIdx.x > 23) {
            printf("i %d value[%d] %x\n", i, i/32, value);
        }
#endif
        if (found_one) {
            mnt4g1::p_double(mod, mod, rx, ry, rz, rx, ry, rz);
#if 0
            if (threadIdx.x > 23) {
            printf("double result\n");
            dump(rx, 24);
            }
#endif
        }
        if ((value)&(1<<i%32)) {
            if (found_one == false) {
                rx = x1;
                ry = y1;
                rz = z1;
            } else {
                mnt4g1::pq_plus(mod, rx, ry, rz, x1, y1, z1, rx, ry, rz);
            }
#if 0
            if (threadIdx.x > 23) {
            printf("add result\n");
            dump(rx, 24);
            }
#endif
            found_one = true;
        }
        i --;
        count ++;
#if 0
        if (threadIdx.x > 23) {
        //if (count >20) break;
        }
#endif
    }
    x3 = rx;
    y3 = ry;
    z3 = rz;
#if 0
    printf("final result\n");
    dump(x3, 24);
    dump(y3, 24);
    dump(z3, 24);
#endif
  }
};

template< typename fixnum >
struct mnt4g2_calc_np {
    __device__ void operator()(fixnum mod, fixnum w, fixnum x10, fixnum x11, fixnum y10, fixnum y11, fixnum z10, fixnum z11, fixnum &x30, fixnum &x31, fixnum &y30, fixnum &y31, fixnum &z30, fixnum &z31) {
    typedef modnum_monty_cios<fixnum> modnum;
    typedef mnt4_g2<fixnum> mnt4g2;
    modnum m(mod);
    fixnum tempw = w;
    int i = 24*32 - 1;
    bool found_one = false;
    int count = 0;
    while(i >= 0) {
        size_t value = fixnum::get(tempw, i/32);
        //printf("value[%d] is %x\n", i, value);
        if (found_one) {
            mnt4g2::p_double(mod, x30, x31, y30, y31, z30, z31, x30, x31, y30, y31, z30, z31);
        }
        if ((value)&(1<<i%32)) {
            if (found_one == false) {
                x30 = x10;
                x31 = x11;
                y30 = y10;
                y31 = y11;
                z30 = z10;
                z31 = z11;
            } else {
                mnt4g2::pq_plus(mod, x30, x31, y30, y31, z30, z31, x10, x11, y10, y11, z10, z11, x30, x31, y30, y31, z30, z31);
            }
            found_one = true;
        }
        i --;
        count ++;
        //if (count >= 50) break;
    }
  }
};

template< typename fixnum >
struct mnt6g1_calc_np {
    __device__ void operator()(fixnum mod, fixnum w, fixnum x1, fixnum y1, fixnum z1, fixnum &x3, fixnum &y3, fixnum &z3) {
    typedef modnum_monty_cios<fixnum> modnum;
    typedef mnt6_g1<fixnum> mnt6g1;
    modnum m(mod);
    fixnum rx, ry, rz;
    fixnum tempw = w;
    int i = 24*32 - 1;
    bool found_one = false;
    int count = 0;
    while(i >= 0) {
        size_t value = fixnum::get(tempw, i/32);
        if (found_one) {
            mnt6g1::p_double(mod, mod, rx, ry, rz, rx, ry, rz);
        }
        if ((value)&(1<<i%32)) {
            if (found_one == false) {
                rx = x1;
                ry = y1;
                rz = z1;
            } else {
                mnt6g1::pq_plus(mod, rx, ry, rz, x1, y1, z1, rx, ry, rz);
            }
            found_one = true;
        }
        i --;
        count ++;
    }
    x3 = rx;
    y3 = ry;
    z3 = rz;
  }
};

template< typename fixnum >
struct mnt6g2_calc_np {
    __device__ void operator()(fixnum mod, fixnum w, fixnum x1, fixnum y1, fixnum z1, fixnum &x3, fixnum &y3, fixnum &z3) {
    typedef modnum_monty_cios<fixnum> modnum;
    typedef mnt6_g2<fixnum> mnt6g2;
    modnum m(mod);
    fixnum rx, ry, rz;
    fixnum tempw = w;
    int i = 24*32 - 1;
    bool found_one = false;
    int count = 0;
    while(i >= 0) {
        size_t value = fixnum::get(tempw, i/32);
        if (found_one) {
            mnt6g2::p_double(mod, mod, rx, ry, rz, rx, ry, rz);
        }
        if ((value)&(1<<i%32)) {
            if (found_one == false) {
                rx = x1;
                ry = y1;
                rz = z1;
            } else {
                mnt6g2::pq_plus(mod, rx, ry, rz, x1, y1, z1, rx, ry, rz);
            }
            found_one = true;
        }
        i --;
        count ++;
    }
    x3 = rx;
    y3 = ry;
    z3 = rz;
  }
};


int mnt4_g1_pq_plus(int n, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t* x2, uint8_t* y2, uint8_t* z2, uint8_t *x3, uint8_t *y3, uint8_t *z3) {
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    fixnum_array *x1in, *y1in, *z1in, *x2in, *y2in, *z2in;
    fixnum_array *rx3, *ry3, *rz3;
    int fn_bytes = 96;
    int step_bytes = n * fn_bytes;
    uint8_t *modulus_bytes = new uint8_t[step_bytes];
    // mnt4 q
    for(int i = 0; i < n; i++) {
        memcpy(modulus_bytes + i*MNT_SIZE, mnt4_modulus, MNT_SIZE);
    }
    auto modulus4 = fixnum_array::create(modulus_bytes, step_bytes, MNT_SIZE);
    
    x1in = fixnum_array::create(x1, step_bytes, fn_bytes);
    y1in = fixnum_array::create(y1, step_bytes, fn_bytes);
    z1in = fixnum_array::create(z1, step_bytes, fn_bytes);
    x2in = fixnum_array::create(x2, step_bytes, fn_bytes);
    y2in = fixnum_array::create(y2, step_bytes, fn_bytes);
    z2in = fixnum_array::create(z2, step_bytes, fn_bytes);

    rx3 = fixnum_array::create(n);
    ry3 = fixnum_array::create(n);
    rz3 = fixnum_array::create(n);
    fixnum_array::template map<mnt4g1_pq_plus>(modulus4, x1in, y1in, z1in, x2in, y2in, z2in, rx3, ry3, rz3);

    int size = n; 
    rx3->retrieve_all(x3, step_bytes, &size);
    ry3->retrieve_all(y3, step_bytes, &size);
    rz3->retrieve_all(z3, step_bytes, &size);
   
    delete x1in; 
    delete y1in; 
    delete z1in; 
    delete x2in; 
    delete y2in; 
    delete z2in; 
    delete rx3;
    delete ry3;
    delete rz3;
    delete modulus4;
    delete modulus_bytes;
    return 0;
}

inline void do_sigma(int nelts, int type, uint8_t *x, uint8_t *y, uint8_t *z, uint8_t *rx, uint8_t *ry, uint8_t *rz) {
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    fixnum_array *x1in, *y1in, *z1in, *x2in, *y2in, *z2in;
    fixnum_array *rx3, *ry3, *rz3;

    int half_bytes = MNT_SIZE*nelts/2;
    uint8_t *modulus_bytes = new uint8_t[half_bytes];
    // mnt4 q
    for(int i = 0; i < nelts/2; i++) {
        memcpy(modulus_bytes + i*MNT_SIZE, mnt4_modulus, MNT_SIZE);
    }
    auto modulus4 = fixnum_array::create(modulus_bytes, half_bytes, MNT_SIZE);
    x1in = fixnum_array::create(x, half_bytes, MNT_SIZE);
    y1in = fixnum_array::create(y, half_bytes, MNT_SIZE);
    z1in = fixnum_array::create(z, half_bytes, MNT_SIZE);
    x2in = fixnum_array::create(x + half_bytes, half_bytes, MNT_SIZE);
    y2in = fixnum_array::create(y + half_bytes, half_bytes, MNT_SIZE);
    z2in = fixnum_array::create(z + half_bytes, half_bytes, MNT_SIZE);

    rx3 = fixnum_array::create(nelts/2);
    ry3 = fixnum_array::create(nelts/2);
    rz3 = fixnum_array::create(nelts/2);
    fixnum_array::template map<mnt4g1_pq_plus>(modulus4, x1in, y1in, z1in, x2in, y2in, z2in, rx3, ry3, rz3);
    
    int size = nelts/2;
    rx3->retrieve_all(rx, half_bytes, &size);
    ry3->retrieve_all(ry, half_bytes, &size);
    rz3->retrieve_all(rz, half_bytes, &size);
    delete x1in;
    delete y1in;
    delete z1in;
    delete x2in;
    delete y2in;
    delete z2in;
    delete rx3;
    delete ry3;
    delete rz3;
    delete modulus4;
    delete modulus_bytes;
}

int do_calc_np_sigma(int nelts, uint8_t* scalar, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t *x3, uint8_t *y3, uint8_t *z3) {
    clock_t start = clock();
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    printf("calc do_calc_np_sigma\n");
    printf("nelts %d\n", nelts);
    int step = nelts;
    int size = nelts;
    int DATA_SIZE = 96;
    int fn_bytes = DATA_SIZE;
    int step_bytes = fn_bytes * step;
    uint8_t *x1bytes = x1;
    uint8_t *y1bytes = y1;
    uint8_t *z1bytes = z1;
    uint8_t *x3bytes = new uint8_t[step_bytes];
    uint8_t *y3bytes = new uint8_t[step_bytes];
    uint8_t *z3bytes = new uint8_t[step_bytes];
    fixnum_array *dx3, *dy3, *dz3, *x1in, *y1in, *z1in;
    fixnum_array *x2in, *y2in, *z2in;
    uint8_t *modulus_bytes = new uint8_t[step_bytes];
    // mnt4 q
    memset(modulus_bytes, 0x0, step_bytes);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt4_modulus, fn_bytes);
    }
    auto modulus4 = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // mnt6 q
    memset(modulus_bytes, 0x0, step_bytes);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt6_modulus, fn_bytes);
    }
    auto modulus6 = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // mnt4 a
    memset(modulus_bytes, 0x0, step_bytes);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt4_a, fn_bytes);
    }
    auto mnt4a = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // scaler
    auto modulusw = fixnum_array::create(scalar, step_bytes, fn_bytes);
    
    // sigma result
    fixnum_array *rx3, *ry3, *rz3;
    int got_result = false;

    for (int i = 0; i < nelts; i+=step) {
        dx3 = fixnum_array::create(step);
        dy3 = fixnum_array::create(step);
        dz3 = fixnum_array::create(step);
        x1in = fixnum_array::create(x1bytes, step_bytes, fn_bytes);
        y1in = fixnum_array::create(y1bytes, step_bytes, fn_bytes);
        z1in = fixnum_array::create(z1bytes, step_bytes, fn_bytes);
        fixnum_array::template map<mnt4g1_calc_np>(modulus4, modulusw, x1in, y1in, z1in, dx3, dy3, dz3);

        dx3->retrieve_all(x3bytes, step_bytes, &size);
        dy3->retrieve_all(y3bytes, step_bytes, &size);
        dz3->retrieve_all(z3bytes, step_bytes, &size);
#ifdef PARALLEL_SIGMA
        int start = nelts%2;
        int rnelts = nelts - start;
        uint8_t *rx, *ry, *rz;
        rx = new uint8_t[MNT_SIZE*rnelts/2];
        ry = new uint8_t[MNT_SIZE*rnelts/2];
        rz = new uint8_t[MNT_SIZE*rnelts/2];
        while(rnelts > 1) {
            do_sigma(rnelts, 1, x3bytes + start*MNT_SIZE, y3bytes + start*MNT_SIZE, z3bytes + start*MNT_SIZE, rx, ry, rz);
            rnelts = rnelts >> 1;
            memcpy(x3bytes + start*MNT_SIZE, rx, rnelts*MNT_SIZE);
            memcpy(y3bytes + start*MNT_SIZE, ry, rnelts*MNT_SIZE);
            memcpy(z3bytes + start*MNT_SIZE, rz, rnelts*MNT_SIZE);
            if (rnelts > 1 && rnelts%2) {
                if (start == 0) {
                    start = 1;
                    rnelts -= 1;
                } else {
                    start = 0;
                    rnelts += 1;
                }
            }
        }
        delete rx;
        delete ry;
        delete rz;
        if (start == 1) {
            // add the first element
            x2in = fixnum_array::create(x3bytes, fn_bytes, fn_bytes);
            y2in = fixnum_array::create(y3bytes, fn_bytes, fn_bytes);
            z2in = fixnum_array::create(z3bytes, fn_bytes, fn_bytes);
            rx3 = fixnum_array::create(x3bytes + fn_bytes, fn_bytes, fn_bytes);
            ry3 = fixnum_array::create(y3bytes + fn_bytes, fn_bytes, fn_bytes);
            rz3 = fixnum_array::create(z3bytes + fn_bytes, fn_bytes, fn_bytes);
            fixnum_array::template map<mnt4g1_pq_plus>(modulus4, x2in, y2in, z2in, rx3, ry3, rz3, rx3, ry3, rz3);
            delete x2in;
            delete y2in;
            delete z2in;
        } else {
            memcpy(x3, x3bytes, fn_bytes);
            memcpy(y3, y3bytes, fn_bytes);
            memcpy(z3, z3bytes, fn_bytes);
            got_result = true;
        }
#else
        bool result_set = false;
        // start add from second element
        int start = 1;
        if (i == 0) {
            rx3 = fixnum_array::create(x3bytes + start * fn_bytes, fn_bytes, fn_bytes);
            ry3 = fixnum_array::create(y3bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rz3 = fixnum_array::create(z3bytes + start * fn_bytes, fn_bytes, fn_bytes);
            result_set = true;
        }
        int k = 0;
        if (result_set && i == 0) {
            k = start + 1;
        }
        for (; k < step; k ++)
        {
            x2in = fixnum_array::create(x3bytes + k * fn_bytes, fn_bytes, fn_bytes);
            y2in = fixnum_array::create(y3bytes + k * fn_bytes, fn_bytes, fn_bytes);
            z2in = fixnum_array::create(z3bytes + k * fn_bytes, fn_bytes, fn_bytes);
            fixnum_array::template map<mnt4g1_pq_plus>(modulus4, rx3, ry3, rz3, x2in, y2in, z2in, rx3, ry3, rz3);
            delete x2in;
            delete y2in;
            delete z2in;
        }
        // add the first element
        x2in = fixnum_array::create(x3bytes, fn_bytes, fn_bytes);
        y2in = fixnum_array::create(y3bytes, fn_bytes, fn_bytes);
        z2in = fixnum_array::create(z3bytes, fn_bytes, fn_bytes);
        fixnum_array::template map<mnt4g1_pq_plus>(modulus4, x2in, y2in, z2in, rx3, ry3, rz3, rx3, ry3, rz3);
        delete x2in;
        delete y2in;
        delete z2in;
#endif
        delete x1in;
        delete y1in;
        delete z1in;
        delete dx3;
        delete dy3;
        delete dz3;
    }
    if (!got_result) {
        size = 1;
        rx3->retrieve_all(x3, fn_bytes, &size);
        ry3->retrieve_all(y3, fn_bytes, &size);
        rz3->retrieve_all(z3, fn_bytes, &size);
        delete rx3;
        delete ry3;
        delete rz3;
    }
    delete x3bytes;
    delete y3bytes;
    delete z3bytes;
    delete modulus_bytes;

    printf("final result");
    printf("\nx3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%02x", x3[k]);
    }
    printf("\ny3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%02x", y3[k]);
    }
    printf("\nz3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
       printf("%02x", z3[k]);
    }
    printf("\n");
    clock_t diff = clock() - start;
    printf("cost time %ld\n", diff);
    return 0;
}

inline void do_sigma(int nelts, int type, uint8_t *x0, uint8_t *x1, uint8_t *y0, uint8_t *y1, uint8_t *z0, uint8_t *z1, uint8_t *rx0, uint8_t *rx1, uint8_t *ry0, uint8_t *ry1, uint8_t *rz0, uint8_t *rz1) {
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    fixnum_array *x10in, *x11in, *y10in, *y11in, *z10in, *z11in, *x20in, *x21in, *y20in, *y21in, *z20in, *z21in;
    fixnum_array *rx30, *rx31, *ry30, *ry31, *rz30, *rz31;

    int half_bytes = MNT_SIZE*nelts/2;
    uint8_t *modulus_bytes = new uint8_t[half_bytes];
    // mnt4 q
    for(int i = 0; i < nelts/2; i++) {
        memcpy(modulus_bytes + i*MNT_SIZE, mnt4_modulus, MNT_SIZE);
    }
    auto modulus4 = fixnum_array::create(modulus_bytes, half_bytes, MNT_SIZE);
    x10in = fixnum_array::create(x0, half_bytes, MNT_SIZE);
    x11in = fixnum_array::create(x1, half_bytes, MNT_SIZE);
    y10in = fixnum_array::create(y0, half_bytes, MNT_SIZE);
    y11in = fixnum_array::create(y1, half_bytes, MNT_SIZE);
    z10in = fixnum_array::create(z0, half_bytes, MNT_SIZE);
    z11in = fixnum_array::create(z1, half_bytes, MNT_SIZE);
    x20in = fixnum_array::create(x0 + half_bytes, half_bytes, MNT_SIZE);
    x21in = fixnum_array::create(x1 + half_bytes, half_bytes, MNT_SIZE);
    y20in = fixnum_array::create(y0 + half_bytes, half_bytes, MNT_SIZE);
    y21in = fixnum_array::create(y1 + half_bytes, half_bytes, MNT_SIZE);
    z20in = fixnum_array::create(z0 + half_bytes, half_bytes, MNT_SIZE);
    z21in = fixnum_array::create(z1 + half_bytes, half_bytes, MNT_SIZE);

    rx30 = fixnum_array::create(nelts/2);
    rx31 = fixnum_array::create(nelts/2);
    ry30 = fixnum_array::create(nelts/2);
    ry31 = fixnum_array::create(nelts/2);
    rz30 = fixnum_array::create(nelts/2);
    rz31 = fixnum_array::create(nelts/2);
    fixnum_array::template map<mnt4g2_pq_plus>(modulus4, x10in, x11in, y10in, y11in, z10in, z11in, x20in, x21in, y20in, y21in, z20in, z21in, rx30, rx31, ry30, ry31, rz30, rz31);
    
    int size = nelts/2;
    rx30->retrieve_all(rx0, half_bytes, &size);
    rx31->retrieve_all(rx1, half_bytes, &size);
    ry30->retrieve_all(ry0, half_bytes, &size);
    ry31->retrieve_all(ry1, half_bytes, &size);
    rz30->retrieve_all(rz0, half_bytes, &size);
    rz31->retrieve_all(rz1, half_bytes, &size);
    delete x10in;
    delete x11in;
    delete y10in;
    delete y11in;
    delete z10in;
    delete z11in;
    delete x20in;
    delete x21in;
    delete y20in;
    delete y21in;
    delete z20in;
    delete z21in;
    delete rx30;
    delete rx31;
    delete ry30;
    delete ry31;
    delete rz30;
    delete rz31;
    delete modulus4;
    delete modulus_bytes;
}

void print_g2(uint8_t *x30, uint8_t *x31, uint8_t *y30, uint8_t *y31, uint8_t *z30, uint8_t *z31) {
    int fn_bytes = 96;
    printf("\nx3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%02x", x30[k]);
    }
    printf(",");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%02x", x31[k]);
    }
    printf("\ny3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%02x", y30[k]);
    }
    printf(",");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%02x", y31[k]);
    }
    printf("\nz3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
       printf("%02x", z30[k]);
    }
    printf(",");
    for (int k = fn_bytes-1; k >= 0; k--) {
       printf("%02x", z31[k]);
    }
    printf("\n");

}

int do_calc_np_sigma_mnt4_g2(int nelts, uint8_t * scalar, uint8_t* x10, uint8_t* x11, uint8_t* y10, uint8_t* y11, uint8_t* z10, uint8_t* z11, uint8_t *x30, uint8_t *x31, uint8_t *y30, uint8_t *y31, uint8_t *z30, uint8_t *z31) {
    clock_t start = clock();
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    printf("calc do_calc_np_sigma_mnt4_g2\n");
    printf("nelts %d\n", nelts);
    // test
#if 0
    nelts = 1;
    int sn = 96;
    x10 += sn;
    x11 += sn;
    y10 += sn;
    y11 += sn;
    z10 += sn;
    z11 += sn;
    scalar += sn;
#endif
    int step = nelts;
    int size = nelts;
    int DATA_SIZE = 96;
    int fn_bytes = DATA_SIZE;
    int step_bytes = fn_bytes * step;
    uint8_t *x10bytes = x10;
    uint8_t *x11bytes = x11;
    uint8_t *y10bytes = y10;
    uint8_t *y11bytes = y11;
    uint8_t *z10bytes = z10;
    uint8_t *z11bytes = z11;
    uint8_t *x30bytes = new uint8_t[step_bytes];
    uint8_t *x31bytes = new uint8_t[step_bytes];
    uint8_t *y30bytes = new uint8_t[step_bytes];
    uint8_t *y31bytes = new uint8_t[step_bytes];
    uint8_t *z30bytes = new uint8_t[step_bytes];
    uint8_t *z31bytes = new uint8_t[step_bytes];
    fixnum_array *dx30, *dx31, *dy30, *dy31, *dz30, *dz31, *x10in, *x11in, *y10in, *y11in, *z10in, *z11in;
    fixnum_array *x20in, *x21in, *y20in, *y21in, *z20in, *z21in;
    uint8_t *modulus_bytes = new uint8_t[step_bytes];
    // mnt4 q
    memset(modulus_bytes, 0x0, step_bytes);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt4_modulus, fn_bytes);
    }
    auto modulus4 = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // mnt6 q
    memset(modulus_bytes, 0x0, step_bytes);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt6_modulus, fn_bytes);
    }
    auto modulus6 = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // mnt4 a
    memset(modulus_bytes, 0x0, step_bytes);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt4_a, fn_bytes);
    }
    auto mnt4a = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // scalar
    auto modulusw = fixnum_array::create(scalar, step_bytes, fn_bytes);
    
    // sigma result
    fixnum_array *rx30, *rx31, *ry30, *ry31, *rz30, *rz31;
    int got_result = false;

#if 0
    printf("x10:\n");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", x10bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nx11:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", x11bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\ny10:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", y10bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\ny11:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", y11bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nz10:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", z10bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nz11:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", z11bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nscalar:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", scalar[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\n");
#endif
    for (int i = 0; i < nelts; i+=step) {
        dx30 = fixnum_array::create(step);
        dx31 = fixnum_array::create(step);
        dy30 = fixnum_array::create(step);
        dy31 = fixnum_array::create(step);
        dz30 = fixnum_array::create(step);
        dz31 = fixnum_array::create(step);
        x10in = fixnum_array::create(x10bytes, step_bytes, fn_bytes);
        x11in = fixnum_array::create(x11bytes, step_bytes, fn_bytes);
        y10in = fixnum_array::create(y10bytes, step_bytes, fn_bytes);
        y11in = fixnum_array::create(y11bytes, step_bytes, fn_bytes);
        z10in = fixnum_array::create(z10bytes, step_bytes, fn_bytes);
        z11in = fixnum_array::create(z11bytes, step_bytes, fn_bytes);
        fixnum_array::template map<mnt4g2_calc_np>(modulus4, modulusw, x10in, x11in, y10in, y11in, z10in, z11in, dx30, dx31, dy30, dy31, dz30, dz31);

        dx30->retrieve_all(x30bytes, step_bytes, &size);
        dx31->retrieve_all(x31bytes, step_bytes, &size);
        dy30->retrieve_all(y30bytes, step_bytes, &size);
        dy31->retrieve_all(y31bytes, step_bytes, &size);
        dz30->retrieve_all(z30bytes, step_bytes, &size);
        dz31->retrieve_all(z31bytes, step_bytes, &size);
        delete x10in;
        delete x11in;
        delete y10in;
        delete y11in;
        delete z10in;
        delete z11in;
        delete dx30;
        delete dx31;
        delete dy30;
        delete dy31;
        delete dz30;
        delete dz31;
#ifdef PARALLEL_SIGMA
        int start = nelts%2;
        int rnelts = nelts - start;
        uint8_t *rx0, *rx1, *ry0, *ry1, *rz0, *rz1;
        rx0 = new uint8_t[MNT_SIZE*rnelts/2];
        rx1 = new uint8_t[MNT_SIZE*rnelts/2];
        ry0 = new uint8_t[MNT_SIZE*rnelts/2];
        ry1 = new uint8_t[MNT_SIZE*rnelts/2];
        rz0 = new uint8_t[MNT_SIZE*rnelts/2];
        rz1 = new uint8_t[MNT_SIZE*rnelts/2];
        while(rnelts > 1) {
            do_sigma(rnelts, 1, x30bytes + start*MNT_SIZE, x31bytes + start*MNT_SIZE, y30bytes + start*MNT_SIZE, y31bytes + start*MNT_SIZE, z30bytes + start*MNT_SIZE, z31bytes + start*MNT_SIZE, rx0, rx1, ry0, ry1, rz0, rz1);
            rnelts = rnelts >> 1;
            memcpy(x30bytes + start*MNT_SIZE, rx0, rnelts*MNT_SIZE);
            memcpy(x31bytes + start*MNT_SIZE, rx1, rnelts*MNT_SIZE);
            memcpy(y30bytes + start*MNT_SIZE, ry0, rnelts*MNT_SIZE);
            memcpy(y31bytes + start*MNT_SIZE, ry1, rnelts*MNT_SIZE);
            memcpy(z30bytes + start*MNT_SIZE, rz0, rnelts*MNT_SIZE);
            memcpy(z31bytes + start*MNT_SIZE, rz1, rnelts*MNT_SIZE);
            if (rnelts > 1 && rnelts%2) {
                if (start == 0) {
                    start = 1;
                    rnelts -= 1;
                } else {
                    start = 0;
                    rnelts += 1;
                }
            }
        }
        delete rx0;
        delete rx1;
        delete ry0;
        delete ry1;
        delete rz0;
        delete rz1;
        if (start == 1) {
            // add the first element
            x20in = fixnum_array::create(x30bytes, fn_bytes, fn_bytes);
            x21in = fixnum_array::create(x31bytes, fn_bytes, fn_bytes);
            y20in = fixnum_array::create(y30bytes, fn_bytes, fn_bytes);
            y21in = fixnum_array::create(y31bytes, fn_bytes, fn_bytes);
            z20in = fixnum_array::create(z30bytes, fn_bytes, fn_bytes);
            z21in = fixnum_array::create(z31bytes, fn_bytes, fn_bytes);
            rx30 = fixnum_array::create(x30bytes + fn_bytes, fn_bytes, fn_bytes);
            rx31 = fixnum_array::create(x31bytes + fn_bytes, fn_bytes, fn_bytes);
            ry30 = fixnum_array::create(y30bytes + fn_bytes, fn_bytes, fn_bytes);
            ry31 = fixnum_array::create(y31bytes + fn_bytes, fn_bytes, fn_bytes);
            rz30 = fixnum_array::create(z30bytes + fn_bytes, fn_bytes, fn_bytes);
            rz31 = fixnum_array::create(z31bytes + fn_bytes, fn_bytes, fn_bytes);
            fixnum_array::template map<mnt4g2_pq_plus>(modulus4, rx30, rx31, ry30, ry31, rz30, rz31, x20in, x21in, y20in, y21in, z20in, z21in, rx30, rx31, ry30, ry31, rz30, rz31);
            delete x20in;
            delete x21in;
            delete y20in;
            delete y21in;
            delete z20in;
            delete z21in;
        } else {
            memcpy(x30, x30bytes, fn_bytes);
            memcpy(x31, x31bytes, fn_bytes);
            memcpy(y30, y30bytes, fn_bytes);
            memcpy(y31, y31bytes, fn_bytes);
            memcpy(z30, z30bytes, fn_bytes);
            memcpy(z31, z31bytes, fn_bytes);
            got_result = true;
        }
#else
        bool result_set = false;
        // start add from second element
        int start = 1;
        if (i == 0) {
            //print_g2(x30bytes + start * fn_bytes, x31bytes + start * fn_bytes, y30bytes + start * fn_bytes, y31bytes + start * fn_bytes, z30bytes + start * fn_bytes, z31bytes + start * fn_bytes);
            rx30 = fixnum_array::create(x30bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rx31 = fixnum_array::create(x31bytes + start * fn_bytes, fn_bytes, fn_bytes);
            ry30 = fixnum_array::create(y30bytes + start * fn_bytes, fn_bytes, fn_bytes);
            ry31 = fixnum_array::create(y31bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rz30 = fixnum_array::create(z30bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rz31 = fixnum_array::create(z31bytes + start * fn_bytes, fn_bytes, fn_bytes);
            result_set = true;
        }
        int k = 0;
        if (result_set && i == 0) {
            k = start + 1;
        }
        for (; k < step; k ++)
        {
            //print_g2(x30bytes + k * fn_bytes, x31bytes + k * fn_bytes, y30bytes + k * fn_bytes, y31bytes + k * fn_bytes, z30bytes + k * fn_bytes, z31bytes + k * fn_bytes);
            x20in = fixnum_array::create(x30bytes + k * fn_bytes, fn_bytes, fn_bytes);
            x21in = fixnum_array::create(x31bytes + k * fn_bytes, fn_bytes, fn_bytes);
            y20in = fixnum_array::create(y30bytes + k * fn_bytes, fn_bytes, fn_bytes);
            y21in = fixnum_array::create(y31bytes + k * fn_bytes, fn_bytes, fn_bytes);
            z20in = fixnum_array::create(z30bytes + k * fn_bytes, fn_bytes, fn_bytes);
            z21in = fixnum_array::create(z31bytes + k * fn_bytes, fn_bytes, fn_bytes);
            fixnum_array::template map<mnt4g2_pq_plus>(modulus4, rx30, rx31, ry30, ry31, rz30, rz31, x20in, x21in, y20in, y21in, z20in, z21in, rx30, rx31, ry30, ry31, rz30, rz31);
            delete x20in;
            delete x21in;
            delete y20in;
            delete y21in;
            delete z20in;
            delete z21in;
#if 0
        rx30->retrieve_all(x30, fn_bytes, &size);
        rx31->retrieve_all(x31, fn_bytes, &size);
        ry30->retrieve_all(y30, fn_bytes, &size);
        ry31->retrieve_all(y31, fn_bytes, &size);
        rz30->retrieve_all(z30, fn_bytes, &size);
        rz31->retrieve_all(z31, fn_bytes, &size);
            print_g2(x30, x31, y30, y31, z30, z31);
#endif
        }
        // add the first element
        x20in = fixnum_array::create(x30bytes, fn_bytes, fn_bytes);
        x21in = fixnum_array::create(x31bytes, fn_bytes, fn_bytes);
        y20in = fixnum_array::create(y30bytes, fn_bytes, fn_bytes);
        y21in = fixnum_array::create(y31bytes, fn_bytes, fn_bytes);
        z20in = fixnum_array::create(z30bytes, fn_bytes, fn_bytes);
        z21in = fixnum_array::create(z31bytes, fn_bytes, fn_bytes);
        fixnum_array::template map<mnt4g2_pq_plus>(modulus4, rx30, rx31, ry30, ry31, rz30, rz31, x20in, x21in, y20in, y21in, z20in, z21in, rx30, rx31, ry30, ry31, rz30, rz31);
        delete x20in;
        delete x21in;
        delete y20in;
        delete y21in;
        delete z20in;
        delete z21in;
#endif
    }
    if (!got_result) {
        size = 1;
        rx30->retrieve_all(x30, fn_bytes, &size);
        rx31->retrieve_all(x31, fn_bytes, &size);
        ry30->retrieve_all(y30, fn_bytes, &size);
        ry31->retrieve_all(y31, fn_bytes, &size);
        rz30->retrieve_all(z30, fn_bytes, &size);
        rz31->retrieve_all(z31, fn_bytes, &size);
        delete rx30;
        delete rx31;
        delete ry30;
        delete ry31;
        delete rz30;
        delete rz31;
    }
    delete x30bytes;
    delete x31bytes;
    delete y30bytes;
    delete y31bytes;
    delete z30bytes;
    delete z31bytes;
    delete modulus_bytes;

    printf("mnt4_g2 final result");
    print_g2(x30, x31, y30, y31, z30, z31);
    clock_t diff = clock() - start;
    printf("cost time %ld\n", diff);
    return 0;
}
