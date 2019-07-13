#include <cstdio>
#include <vector>

#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_init.hpp>
#include "cuda-fixnum/main.cuh"

using namespace libff;

Fq<mnt6753_pp> read_mnt6_fq(FILE* input) {
  // bigint<mnt6753_q_limbs> n;
  Fq<mnt6753_pp> x;
  fread((void *) x.mont_repr.data, libff::mnt6753_q_limbs * sizeof(mp_size_t), 1, input);
  return x;
}

void write_mnt6_fq(FILE* output, Fq<mnt6753_pp> x) {
  fwrite((void *) x.mont_repr.data, libff::mnt6753_q_limbs * sizeof(mp_size_t), 1, output);
}

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

// The actual code for doing Fq2 multiplication lives in libff/algebra/fields/fp2.tcc
int main(int argc, char *argv[])
{
    // argv should be
    // { "main", "compute" or "compute-numeral", inputs, outputs }

    mnt6753_pp::init_public_params();

    auto inputs = fopen(argv[2], "r");
    auto outputs = fopen(argv[3], "w");

    auto read_mnt6 = read_mnt6_fq;
    auto write_mnt6 = write_mnt6_fq;
    auto read_mnt6_q3 = read_mnt6_fq3;
    auto write_mnt6_q3 = write_mnt6_fq3;

    while (true) {
      size_t n;
      size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);

      if (elts_read == 0) { break; }
      size_t data_size = n * 96;

      uint8_t *x1buf = new uint8_t[data_size];
      uint8_t *x2buf = new uint8_t[data_size];
      uint8_t *x3buf = new uint8_t[data_size];
      uint8_t *y1buf = new uint8_t[data_size];
      uint8_t *y2buf = new uint8_t[data_size];
      uint8_t *y3buf = new uint8_t[data_size];
      uint8_t *out1buf = new uint8_t[data_size];
      uint8_t *out2buf = new uint8_t[data_size];
      uint8_t *out3buf = new uint8_t[data_size];
      size_t offset = 0;
      for (int i = 0; i < n; i++) {
          fread(x1buf + offset, 96, 1, inputs); 
          fread(x2buf + offset, 96, 1, inputs); 
          fread(x3buf + offset, 96, 1, inputs); 
          offset += 96;
      }
      offset = 0;
      for (int i = 0; i < n; i++) {
          fread(y1buf + offset, 96, 1, inputs); 
          fread(y2buf + offset, 96, 1, inputs); 
          fread(y3buf + offset, 96, 1, inputs); 
          offset += 96;
      }

      mnt6_g2_mul(n, x1buf, x2buf, x3buf, y1buf, y2buf, y3buf, out1buf, out2buf, out3buf);

      offset = 0;
      for (int i = 0; i < n; i++) {
          fwrite(out1buf + offset, 96, 1, outputs);
          fwrite(out2buf + offset, 96, 1, outputs);
          fwrite(out3buf + offset, 96, 1, outputs);
          offset += 96;
      }
      delete x1buf;
      delete x2buf;
      delete x3buf;
      delete y1buf;
      delete y2buf;
      delete y3buf;
      delete out1buf;
      delete out2buf;
      delete out3buf;
    }
    fclose(outputs);


    return 0;
}
