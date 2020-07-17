#include "immintrin.h"
#include <iostream>
#include <iomanip>
struct Complex_int16 {
    int16_t real;
    int16_t imag;
};

void print_m512i(__m512i v) {
    int16_t* val = (int16_t*)&v;
    std::cout << "__m512i: ";
    for(int i = 0; i < 32; i+=2) {
        std::cout << "(" << std::setw(3) << val[i] << "," << std::setw(3) << val[i+1] << "), ";
    }
    std::cout << std::endl;
}

int main() {
    int16_t t[32] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,20,31,32};
    __m512i test = _mm512_loadu_si512((const void*)t);
    print_m512i(test);
    __m512i res = _mm512_shufflehi_epi16(test, 0xB1);
    res = _mm512_shufflelo_epi16(res, 0xB1); // one shufhi and shuflo will swap pairs
    print_m512i(res);
}