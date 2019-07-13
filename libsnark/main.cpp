#include <cstdio>
#include <vector>

#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>
#include "cuda-fixnum/main.cuh"

using namespace libff;

void setBigData(bigint<mnt4753_q_limbs> *bigintV, uint8_t *val, int size) {
    for (int i = 0; i < size; i+=8) {
        unsigned long x0, x1, x2, x3, x4, x5, x6, x7;
        x0 = val[i];
        x1 = val[i+1];
        x2 = val[i+2];
        x3 = val[i+3];
        x4 = val[i+4];
        x5 = val[i+5];
        x6 = val[i+6];
        x7 = val[i+7];
        bigintV->data[i/8] = x7<<56 | x6<<48 | x5<<40 | x4<<32 | x3<<24 | x2<<16 | x1<<8 | x0;
    }
}

// mnt4 montgomery
Fq<mnt4753_pp> read_mnt4_fq(FILE* input) {
  Fq<mnt4753_pp> x;
  fread((void *) x.mont_repr.data, libff::mnt4753_q_limbs * sizeof(mp_size_t), 1, input);
  return x;
}

void write_mnt4_fq(FILE* output, Fq<mnt4753_pp> x) {
  fwrite((void *) x.mont_repr.data, libff::mnt4753_q_limbs * sizeof(mp_size_t), 1, output);
}

// mnt6 montgomery
Fq<mnt6753_pp> read_mnt6_fq(FILE* input) {
  Fq<mnt6753_pp> x;
  fread((void *) x.mont_repr.data, libff::mnt6753_q_limbs * sizeof(mp_size_t), 1, input);
  return x;
}

void write_mnt6_fq(FILE* output, Fq<mnt6753_pp> x) {
  fwrite((void *) x.mont_repr.data, libff::mnt6753_q_limbs * sizeof(mp_size_t), 1, output);
}

// mnt4 fq2 montgomery
Fqe<mnt4753_pp> read_mnt4_fq2(FILE* input) {
  Fq<mnt4753_pp> c0 = read_mnt4_fq(input);
  Fq<mnt4753_pp> c1 = read_mnt4_fq(input);
  return Fqe<mnt4753_pp>(c0, c1);
}

void write_mnt4_fq2(FILE* output, Fqe<mnt4753_pp> x) {
  write_mnt4_fq(output, x.c0);
  write_mnt4_fq(output, x.c1);
}

// mnt6 fq3 montgomery
Fqe<mnt6753_pp> read_mnt6_fq3(FILE* input) {
  Fq<mnt6753_pp> c0 = read_mnt6_fq(input);
  Fq<mnt6753_pp> c1 = read_mnt6_fq(input);
  Fq<mnt6753_pp> c2 = read_mnt6_fq(input);
  return Fqe<mnt6753_pp>(c0, c1, c2);
}

void write_mnt6_fq3(FILE* output, Fqe<mnt6753_pp> x) {
  write_mnt6_fq(output, x.c0);
  write_mnt6_fq(output, x.c1);
  write_mnt6_fq(output, x.c2);
}

// mnt4 groups montgomery
G1<mnt4753_pp> read_mnt4_g1(FILE* input) {
  Fq<mnt4753_pp> x = read_mnt4_fq(input);
  Fq<mnt4753_pp> y = read_mnt4_fq(input);
  return G1<mnt4753_pp>(x, y, Fq<mnt4753_pp>::one());
}

void write_mnt4_g1(FILE* output, G1<mnt4753_pp> g) {
  g.to_affine_coordinates();
  write_mnt4_fq(output, g.X());
  write_mnt4_fq(output, g.Y());
}

G2<mnt4753_pp> read_mnt4_g2(FILE* input) {
  Fqe<mnt4753_pp> x = read_mnt4_fq2(input);
  Fqe<mnt4753_pp> y = read_mnt4_fq2(input);
  return G2<mnt4753_pp>(x, y, Fqe<mnt4753_pp>::one());
}

void write_mnt4_g2(FILE* output, G2<mnt4753_pp> g) {
  g.to_affine_coordinates();
  write_mnt4_fq2(output, g.X());
  write_mnt4_fq2(output, g.Y());
}

// mnt6 groups montgomery
G1<mnt6753_pp> read_mnt6_g1(FILE* input) {
  Fq<mnt6753_pp> x = read_mnt6_fq(input);
  Fq<mnt6753_pp> y = read_mnt6_fq(input);
  return G1<mnt6753_pp>(x, y, Fq<mnt6753_pp>::one());
}

void write_mnt6_g1(FILE* output, G1<mnt6753_pp> g) {
  g.to_affine_coordinates();
  write_mnt6_fq(output, g.X());
  write_mnt6_fq(output, g.Y());
}

G2<mnt6753_pp> read_mnt6_g2(FILE* input) {
  Fqe<mnt6753_pp> x = read_mnt6_fq3(input);
  Fqe<mnt6753_pp> y = read_mnt6_fq3(input);
  return G2<mnt6753_pp>(x, y, Fqe<mnt6753_pp>::one());
}

void write_mnt6_g2(FILE* output, G2<mnt6753_pp> g) {
  g.to_affine_coordinates();
  write_mnt6_fq3(output, g.X());
  write_mnt6_fq3(output, g.Y());
}

// The actual code for doing Fq2 multiplication lives in libff/algebra/fields/fp2.tcc
int main(int argc, char *argv[])
{
    // argv should be
    // { "main", "compute" or "compute-numeral", inputs, outputs }

    mnt4753_pp::init_public_params();
    mnt6753_pp::init_public_params();

    auto inputs = fopen(argv[2], "r");
    auto outputs = fopen(argv[3], "w");

    // mnt4
    auto read_mnt4 = read_mnt4_fq;
    auto write_mnt4 = write_mnt4_fq;
    auto read_mnt4_q2 = read_mnt4_fq2;
    auto write_mnt4_q2 = write_mnt4_fq2;
    // mnt6
    auto read_mnt6 = read_mnt6_fq;
    auto write_mnt6 = write_mnt6_fq;
    auto read_mnt6_q3 = read_mnt6_fq3;
    auto write_mnt6_q3 = write_mnt6_fq3;
    // mnt4 groups
    auto write_mnt4g1 = write_mnt4_g1;
    auto read_mnt4g1 = read_mnt4_g1;
    auto write_mnt4g2 = write_mnt4_g2;
    auto read_mnt4g2 = read_mnt4_g2;
    // mnt6 groups
    auto read_mnt6g1 = read_mnt6_g1;
    auto write_mnt6g1 = write_mnt6_g1;
    auto read_mnt6g2 = read_mnt6_g2;
    auto write_mnt6g2 = write_mnt6_g2;

    while (true) {
      size_t n;
      size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);

      if (elts_read == 0) { break; }
      size_t esize = 96;
      size_t data_size = n * esize;

      uint8_t *x1buf = new uint8_t[data_size];
      uint8_t *x2buf = new uint8_t[data_size];
      uint8_t *x3buf = new uint8_t[data_size];
      uint8_t *y1buf = new uint8_t[data_size];
      uint8_t *y2buf = new uint8_t[data_size];
      uint8_t *y3buf = new uint8_t[data_size];
      uint8_t *z1buf = new uint8_t[data_size];
      uint8_t *z2buf = new uint8_t[data_size];
      uint8_t *z3buf = new uint8_t[data_size];
      uint8_t *outxbuf = new uint8_t[3*esize];  //3 for 6_g2
      uint8_t *outybuf = new uint8_t[3*esize];
      uint8_t *outzbuf = new uint8_t[3*esize];

      size_t offset = 0;
      for (int i = 0; i < n; i++) {
          fread(x1buf + offset, esize, 1, inputs); 
          offset += esize;
      }
      offset = 0;
      for (int i = 0; i < n; i++) {
          fread(y1buf + offset, esize, 1, inputs); 
          offset += esize;
      }
      offset = 0;
      for (int i = 0; i < n; i++) {
          memcpy(z1buf + offset, Fq<mnt4753_pp>::one().mont_repr.data, esize);
          offset += esize;
      }

      mnt4_g1_sigma(n, x1buf, y1buf, z1buf, outxbuf, outybuf, outzbuf);

      bigint<mnt4753_q_limbs> bigint_x, bigint_y, bigint_z;
      setBigData(&bigint_x, outxbuf, esize);
      setBigData(&bigint_y, outybuf, esize);
      setBigData(&bigint_z, outzbuf, esize);

      G1<mnt4753_pp> g_41 = G1<mnt4753_pp>(bigint_x, bigint_y, bigint_z);
      write_mnt4_g1(outputs, g_41);
      
      delete x1buf;
      delete x2buf;
      delete x3buf;
      delete y1buf;
      delete y2buf;
      delete y3buf;
      delete outxbuf;
      delete outybuf;
      delete outzbuf;
    }
    fclose(outputs);


    return 0;
}
