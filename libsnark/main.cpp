#include <cstdio>
#include <vector>

#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_init.hpp>
#include "cuda-fixnum/main.cuh"

using namespace libff;

Fq<mnt4753_pp> read_mnt4_fq(FILE* input) {
  // bigint<mnt4753_q_limbs> n;
  Fq<mnt4753_pp> x;
  fread((void *) x.mont_repr.data, libff::mnt4753_q_limbs * sizeof(mp_size_t), 1, input);
  return x;
}

void write_mnt4_fq(FILE* output, Fq<mnt4753_pp> x) {
  fwrite((void *) x.mont_repr.data, libff::mnt4753_q_limbs * sizeof(mp_size_t), 1, output);
}

Fqe<mnt4753_pp> read_mnt4_fq2(FILE* input) {
  Fq<mnt4753_pp> c0 = read_mnt4_fq(input);
  Fq<mnt4753_pp> c1 = read_mnt4_fq(input);
  return Fqe<mnt4753_pp>(c0, c1);
}

void write_mnt4_fq2(FILE* output, Fqe<mnt4753_pp> x) {
  write_mnt4_fq(output, x.c0);
  write_mnt4_fq(output, x.c1);
}

// The actual code for doing Fq2 multiplication lives in libff/algebra/fields/fp2.tcc
int main(int argc, char *argv[])
{
    // argv should be
    // { "main", "compute" or "compute-numeral", inputs, outputs }

    mnt4753_pp::init_public_params();

    auto inputs = fopen(argv[2], "r");
    auto outputs = fopen(argv[3], "w");

    auto read_mnt4 = read_mnt4_fq;
    auto write_mnt4 = write_mnt4_fq;
    auto read_mnt4_q2 = read_mnt4_fq2;
    auto write_mnt4_q2 = write_mnt4_fq2;

    while (true) {
      size_t n;
      size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);

      if (elts_read == 0) { break; }
      size_t data_size = n * 96;

      uint8_t *x1buf = new uint8_t[data_size];
      uint8_t *x2buf = new uint8_t[data_size];
      uint8_t *y1buf = new uint8_t[data_size];
      uint8_t *y2buf = new uint8_t[data_size];
      uint8_t *out1buf = new uint8_t[data_size];
      uint8_t *out2buf = new uint8_t[data_size];
      size_t offset = 0;
      for (int i = 0; i < n; i++) {
          fread(x1buf + offset, 96, 1, inputs); 
          fread(x2buf + offset, 96, 1, inputs); 
          offset += 96;
      }
      offset = 0;
      for (int i = 0; i < n; i++) {
          fread(y1buf + offset, 96, 1, inputs); 
          fread(y2buf + offset, 96, 1, inputs); 
          offset += 96;
      }

      mnt4_g2_mul(n, x1buf, x2buf, y1buf, y2buf, out1buf, out2buf);

      offset = 0;
      for (int i = 0; i < n; i++) {
          fwrite(out1buf + offset, 96, 1, outputs);
          fwrite(out2buf + offset, 96, 1, outputs);
          offset += 96;
      }
      delete x1buf;
      delete x2buf;
      delete y1buf;
      delete y2buf;
      delete out1buf;
      delete out2buf;
    }
    fclose(outputs);


    return 0;
}
