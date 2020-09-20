#ifndef BYL_JIT_INT16_MATVEC
#define BYL_JIT_INT16_MATVEC
#include <stdint.h>
#include <xbyak/xbyak.h>
#include <Zydis/Zydis.h>
#include <fstream>
#include <string.h>
#include <iostream>
#include "immintrin.h"
#include <complex>
#include "mkl.h"
#include "timer.hpp"
#include <iomanip>
#include <vector>
#include <assert.h>
#include <numeric>
#include <math.h>
#include <random>

#define DIE(...) { fprintf(stderr, __VA_ARGS__); exit(1); }
#define FIXED_POINT_FRACTIONAL_BITS 9
#define TOLERANCE 0.05f

struct Complex_int16 {
    int16_t real;
    int16_t imag;
    Complex_int16& operator+(const Complex_int16& rhs){ 
        real += rhs.real;
        imag += rhs.imag;
        return *this;
    }
    bool operator==(const Complex_int16& rhs) const { 
        return (real != rhs.real) | (imag != rhs.imag) ? false : true;
    }
    bool operator!=(const Complex_int16& rhs) const { 
        return !(*this == rhs);
    }
    friend std::ostream& operator<<(std::ostream& os, const Complex_int16& c) {
        os << "(" << c.real << "," << c.imag << ")";
        return os;
    }
};
typedef void (*byl_matvec_jit_kernel_t)(const Complex_int16*,  const Complex_int16*,  Complex_int16*);

byl_matvec_jit_kernel_t byl_jit_create_matvec(long m, long k, void** jitter);
void byl_jit_destroy_matvec(byl_matvec_jit_kernel_t);

inline int16_t floatToFixed(float flo) {
    return (int16_t)(round(flo * (1 << FIXED_POINT_FRACTIONAL_BITS)));
}

inline float fixedToFloat(int16_t fix) {
    return ((float)fix / (float)(1 << FIXED_POINT_FRACTIONAL_BITS));
}
#endif