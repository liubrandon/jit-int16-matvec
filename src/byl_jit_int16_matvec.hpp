#ifndef BYL_JIT_INT16_MATVEC
#define BYL_JIT_INT16_MATVEC
#include <stdint.h>
#include <xbyak/xbyak.h>
#include <fstream>
#include <string.h>
#include <iostream>
#include <complex>
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

inline int16_t floatToFixed(float flo) {
    return (int16_t)(round(flo * (1 << FIXED_POINT_FRACTIONAL_BITS)));
}

inline float fixedToFloat(int16_t fix) {
    return ((float)fix / (float)(1 << FIXED_POINT_FRACTIONAL_BITS));
}

struct JitInt16MatVec : Xbyak::CodeGenerator {
    JitInt16MatVec(int m, int k)//, void* userPtr = 0, size_t size = Xbyak::DEFAULT_MAX_CODE_SIZE)
        : Xbyak::CodeGenerator(4096, Xbyak::AutoGrow) // Use Read/Exec mode for security
    {  // Input parameters: rdi=mat, rsi=vec, rdx=res, rcx=, r8=, r9=
        Xbyak::Label broad16,swapPairs,swapAfterPack; // Constant data / "magic numbers" at the end of the struct
        int rowsize = m*4;
        sub(rsp, 0x04*k); // allocate vec size onto stack
        vpbroadcastd(zmm31, dword [rip+broad16]);
        vmovdqu16(zmm0, zword [rip+swapPairs]);      // zmm0 = swapPairs
        // Swap pairs of vector and store that copy on the stack
        for(int i = 0; i < k/16; i++) {
            vmovdqu16(zmm1, zword [rsi+(i*0x40)]);
            vpshufb(zmm1, zmm1, zmm0); 
            vmovdqu16(zword [rsp+(i*0x40)], zmm1);
        }
        // Special kernel for 16x64
        if(m == 16 && k == 64) {
            vxorps(zmm29,zmm29,zmm29); vxorps(zmm28,zmm28,zmm28);
            vxorps(zmm1,zmm1,zmm1);    vxorps(zmm2,zmm2,zmm2);
            vxorps(zmm3,zmm3,zmm3);    vxorps(zmm4,zmm4,zmm4);
            vxorps(zmm21,zmm21,zmm21); vxorps(zmm22,zmm22,zmm22);
            for(int i = 0; i < k; i+=4) {
                vmovdqu16(zmm30, zword [rdi+(rowsize*i)]);
                vmovdqu16(zmm27, zword [rdi+(rowsize*(i+1))]);
                vmovdqu16(zmm26, zword [rdi+(rowsize*(i+2))]);
                vmovdqu16(zmm25, zword [rdi+(rowsize*(i+3))]);

                vpbroadcastd( zmm5, dword [rsi+(0x04*i)]);
                vpmullw(zmm5 , zmm5 , zmm31);
                vpbroadcastd( zmm6, dword [rsp+(0x04*i)]);
                vpdpwssds(zmm29, zmm30, zmm5);
                vpdpwssds(zmm28, zmm30, zmm6);
            
                vpbroadcastd( zmm7, dword [rsi+(0x04*(i+1))]);
                vpmullw(zmm7 , zmm7 , zmm31);
                vpbroadcastd( zmm8, dword [rsp+(0x04*(i+1))]);
                vpdpwssds(zmm1, zmm27, zmm7);
                vpdpwssds(zmm2, zmm27, zmm8);

                vpbroadcastd( zmm9, dword [rsi+(0x04*(i+2))]);
                vpmullw(zmm9 , zmm9 , zmm31);
                vpbroadcastd(zmm10, dword [rsp+(0x04*(i+2))]);
                vpdpwssds(zmm3, zmm26, zmm9);
                vpdpwssds(zmm4, zmm26, zmm10);

                vpbroadcastd(zmm11, dword [rsi+(0x04*(i+3))]);
                vpmullw(zmm11, zmm11, zmm31);
                vpbroadcastd(zmm12, dword [rsp+(0x04*(i+3))]); 
                vpdpwssds(zmm21, zmm25, zmm11);
                vpdpwssds(zmm22, zmm25, zmm12);
            }
            vmovdqu16(zmm0, zword [rip+swapAfterPack]);      // zmm0 = swapAfterPack
            // Sum up accumulators for imaginary results
            vpaddd(zmm22,zmm22,zmm2);
            vpaddd(zmm28,zmm28,zmm4);
            vpaddd(zmm28,zmm28,zmm22);
            // Shift out exccess fractional bits to correct scaling factor
            vpsrad(zmm28, zmm28, FIXED_POINT_FRACTIONAL_BITS);
            // Sum up accumulators for real results
            vpaddd(zmm29,zmm29,zmm3);
            vpaddd(zmm21,zmm21,zmm1);
            vpaddd(zmm29,zmm29,zmm21);
            // Correct scaling factor
            vpsrad(zmm29, zmm29, FIXED_POINT_FRACTIONAL_BITS);
            // Downcast from 32 bits to 16 bits and keep sign, packing both results into zmm29
            vpackssdw(zmm29, zmm29, zmm28);
            // Correct order of the final result
            vpshufb(zmm29, zmm29, zmm0);
            // Write everything to memory 
            vmovdqa64(zword [rdx], zmm29);
        }
        // Other sizes take this kernel
        else {
            int inc = 0; // overwritten in the loop
            for(int r = 0; r < m; r += inc) { // TODO: Fix this loop
                // first 16 elements of column 1
                vmovdqu16(zmm30, zword [rdi+(r*0x04)]); // load first column (a_b)
                vpbroadcastd(zmm5, dword [rsi]); // zmm5 = c_d (broadcast first two values from vec to all locations)
                vpmullw(zmm5, zmm5, zmm31); // zmm5 = c_minus_d (negate every other value in zmm5)
                vpbroadcastd(zmm6, dword [rsp]); // zmm6 = d_c
                vpmaddwd(zmm28, zmm30, zmm6); // zmm28 = imag_res accumulator
                vpmaddwd(zmm29, zmm30, zmm5); // zmm29 = real_res accumulator
                inc = 16;
                if (m-r >= 32) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+0x40]);
                    vpmaddwd(zmm27, zmm30, zmm5);
                    vpmaddwd(zmm26, zmm30, zmm6); inc = 32;
                }
                if (m-r >= 48) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+0x80]);
                    vpmaddwd(zmm25, zmm30, zmm5);
                    vpmaddwd(zmm24, zmm30, zmm6); inc = 48;
                }
                if (m-r >= 64) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+0xC0]);
                    vpmaddwd(zmm23, zmm30, zmm5);
                    vpmaddwd(zmm22, zmm30, zmm6); inc = 64;
                }        
                if (m-r >= 80) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+0x100]);
                    vpmaddwd(zmm21, zmm30, zmm5);
                    vpmaddwd(zmm20, zmm30, zmm6); inc = 80;
                }
                if (m-r >= 96) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+0x140]);
                    vpmaddwd(zmm19, zmm30, zmm5);
                    vpmaddwd(zmm18, zmm30, zmm6); inc = 96;
                }
                if (m-r >= 112) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+0x180]);
                    vpmaddwd(zmm17, zmm30, zmm5);
                    vpmaddwd(zmm16, zmm30, zmm6); inc = 112;
                }        
                if (m-r >= 128) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+0x1C0]);
                    vpmaddwd(zmm15, zmm30, zmm5);
                    vpmaddwd(zmm14, zmm30, zmm6); inc = 128;
                }        
                if (m-r >= 144) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+0x200]);
                    vpmaddwd(zmm13, zmm30, zmm5);
                    vpmaddwd(zmm12, zmm30, zmm6); inc = 144;
                }
                if (m-r >= 160) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+0x240]);
                    vpmaddwd(zmm11, zmm30, zmm5);
                    vpmaddwd(zmm10, zmm30, zmm6); inc = 160;
                }
                if (m-r >= 176) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+0x280]);
                    vpmaddwd(zmm9 , zmm30, zmm5);
                    vpmaddwd(zmm8 , zmm30, zmm6); inc = 176;
                }        
                if (m-r >= 192) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+0x2C0]);
                    vpmaddwd(zmm7 , zmm30, zmm5);
                    vpmaddwd(zmm4 , zmm30, zmm6); inc = 192;
                }        
                if (m-r >= 208) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+0x300]);
                    vpmaddwd(zmm3 , zmm30, zmm5);
                    vpmaddwd(zmm2 , zmm30, zmm6); inc = 208;
                } 
                for(int i = 1; i < k; i++) {
                    vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)]);
                    vpbroadcastd(zmm5, dword [rsi+(0x04*i)]);
                    vpmullw(zmm5, zmm5, zmm31);
                    vpbroadcastd(zmm6, dword [rsp+(0x04*i)]); 
                    vpdpwssds(zmm29, zmm30, zmm5);
                    vpdpwssds(zmm28, zmm30, zmm6);
                    if (m-r >= 32) {
                        vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)+0x40]);
                        vpdpwssds(zmm27, zmm30, zmm5);
                        vpdpwssds(zmm26, zmm30, zmm6);
                    } 
                    if (m-r >= 48) {
                        vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)+0x80]);
                        vpdpwssds(zmm25, zmm30, zmm5);
                        vpdpwssds(zmm24, zmm30, zmm6);
                    } 
                    if (m-r >= 64) {
                        vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)+0xC0]);
                        vpdpwssds(zmm23, zmm30, zmm5);
                        vpdpwssds(zmm22, zmm30, zmm6);
                    }
                    if (m-r >= 80) {
                        vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)+0x100]);
                        vpdpwssds(zmm21, zmm30, zmm5);
                        vpdpwssds(zmm20, zmm30, zmm6);
                    } 
                    if (m-r >= 96) {
                        vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)+0x140]);
                        vpdpwssds(zmm19, zmm30, zmm5);
                        vpdpwssds(zmm18, zmm30, zmm6);
                    } 
                    if (m-r >= 112) {
                        vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)+0x180]);
                        vpdpwssds(zmm17, zmm30, zmm5);
                        vpdpwssds(zmm16, zmm30, zmm6);
                    }
                    if (m-r >= 128) {
                        vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)+0x1C0]);
                        vpdpwssds(zmm15, zmm30, zmm5);
                        vpdpwssds(zmm14, zmm30, zmm6);
                    }
                    if (m-r >= 144) {
                        vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)+0x200]);
                        vpdpwssds(zmm13, zmm30, zmm5);
                        vpdpwssds(zmm12, zmm30, zmm6);
                    } 
                    if (m-r >= 160) {
                        vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)+0x240]);
                        vpdpwssds(zmm11, zmm30, zmm5);
                        vpdpwssds(zmm10, zmm30, zmm6);
                    } 
                    if (m-r >= 176) {
                        vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)+0x280]);
                        vpdpwssds(zmm9 , zmm30, zmm5);
                        vpdpwssds(zmm8 , zmm30, zmm6);
                    }
                    if (m-r >= 192) {
                        vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)+0x2C0]);
                        vpdpwssds(zmm7 , zmm30, zmm5);
                        vpdpwssds(zmm4 , zmm30, zmm6);
                    }
                    if (m-r >= 208) {
                        vmovdqu16(zmm30, zword [rdi+(r*0x04)+(rowsize*i)+0x300]);
                        vpdpwssds(zmm3 , zmm30, zmm5);
                        vpdpwssds(zmm2 , zmm30, zmm6);
                    }
                }
                vmovdqu16(zmm0, zword [rip+swapAfterPack]);      // zmm0 = swapAfterPack
                vpsrad(zmm28, zmm28, FIXED_POINT_FRACTIONAL_BITS);
                vpsrad(zmm29, zmm29, FIXED_POINT_FRACTIONAL_BITS);
                // Downcast from 32 bits to 16 bits and keep sign, packing both results into zmm29
                vpackssdw(zmm29, zmm29, zmm28);
                // Correct order of the final result
                vpshufb(zmm29, zmm29, zmm0);
                // Write to memory
                vmovdqa64(zword [rdx+(r*0x04)], zmm29);
                if (m-r >= 32) {
                    vpsrad(zmm27, zmm27, FIXED_POINT_FRACTIONAL_BITS);
                    vpsrad(zmm26, zmm26, FIXED_POINT_FRACTIONAL_BITS);
                    vpackssdw(zmm27, zmm27, zmm26);
                    vpshufb(zmm27, zmm27, zmm0);
                    vmovdqa64(zword [rdx+(r*0x04)+0x40], zmm27);
                }
                if (m-r >= 48) {
                    vpsrad(zmm25, zmm25, FIXED_POINT_FRACTIONAL_BITS);
                    vpsrad(zmm24, zmm24, FIXED_POINT_FRACTIONAL_BITS);
                    vpackssdw(zmm25, zmm25, zmm24);
                    vpshufb(zmm25, zmm25, zmm0);
                    vmovdqa64(zword [rdx+(r*0x04)+0x80], zmm25);
                }
                if (m-r >= 64) {
                    vpsrad(zmm23, zmm23, FIXED_POINT_FRACTIONAL_BITS);
                    vpsrad(zmm22, zmm22, FIXED_POINT_FRACTIONAL_BITS);
                    vpackssdw(zmm23, zmm23, zmm22);
                    vpshufb(zmm23, zmm23, zmm0);
                    vmovdqa64(zword [rdx+(r*0x04)+0xC0], zmm23);
                }
                if (m-r >= 80) { // TODO: fix here and below with new rotate pack and shuffle instructions 
                    vpslld(zmm20, zmm20, 0x10);
                    vmovdqu16(zmm21 | k1, zmm20);
                    vmovdqa64(zword [rdx+(r*0x04)+0x100], zmm21);
                }
                if (m-r >= 96) {
                    vpslld(zmm18, zmm18, 0x10);
                    vmovdqu16(zmm19 | k1, zmm18);
                    vmovdqa64(zword [rdx+(r*0x04)+0x140], zmm19);
                }
                if (m-r >= 112) {
                    vpslld(zmm16, zmm16, 0x10);
                    vmovdqu16(zmm17 | k1, zmm16);
                    vmovdqa64(zword [rdx+(r*0x04)+0x180], zmm17);
                }
                if (m-r >= 128) {
                    vpslld(zmm14, zmm14, 0x10);
                    vmovdqu16(zmm15 | k1, zmm14);
                    vmovdqa64(zword [rdx+(r*0x04)+0x1C0], zmm15);
                }
                if (m-r >= 144) {
                    vpslld(zmm12, zmm12, 0x10);
                    vmovdqu16(zmm13 | k1, zmm12);
                    vmovdqa64(zword [rdx+(r*0x04)+0x200], zmm13);
                }
                if (m-r >= 160) {
                    vpslld(zmm10, zmm10, 0x10);
                    vmovdqu16(zmm11 | k1, zmm10);
                    vmovdqa64(zword [rdx+(r*0x04)+0x240], zmm11);
                }
                if (m-r >= 176) {
                    vpslld(zmm8, zmm8, 0x10);
                    vmovdqu16(zmm9 | k1, zmm8);
                    vmovdqa64(zword [rdx+(r*0x04)+0x280], zmm9);
                }
                if (m-r >= 192) {
                    vpslld(zmm4, zmm4, 0x10);
                    vmovdqu16(zmm7 | k1, zmm4);
                    vmovdqa64(zword [rdx+(r*0x04)+0x2C0], zmm7);
                }
                if (m-r >= 208) {
                    vpslld(zmm2, zmm2, 0x10);
                    vmovdqu16(zmm3 | k1, zmm2);
                    vmovdqa64(zword [rdx+(r*0x04)+0x300], zmm3);
                }
            }
        }
        add(rsp, 0x04*k); // deallocate vec size from stack
        vzeroupper(); // needed? I think needed if using both AVX2 and other SIMD extensions
        ret();

        // Constant data, to access use [rip + label_name]
        L(broad16);
        dw(1); dw(-1);

        L(swapPairs);
        db(2);db(3);db(0);db(1);db(6);db(7);db(4);db(5);db(10);db(11);db(8);db(9);db(14);db(15);db(12);db(13);db(18);db(19);db(16);db(17);db(22);db(23);db(20);db(21);db(26);db(27);db(24);db(25);db(30);db(31);db(28);db(29);db(34);db(35);db(32);db(33);db(38);db(39);db(36);db(37);db(42);db(43);db(40);db(41);db(46);db(47);db(44);db(45);db(50);db(51);db(48);db(49);db(54);db(55);db(52);db(53);db(58);db(59);db(56);db(57);db(62);db(63);db(60);db(61);
        
        L(swapAfterPack);
        db(0);db(1);db(8);db(9);db(2);db(3);db(10);db(11);db(4);db(5);db(12);db(13);db(6);db(7);db(14);db(15);db(16);db(17);db(24);db(25);db(18);db(19);db(26);db(27);db(20);db(21);db(28);db(29);db(22);db(23);db(30);db(31);db(32);db(33);db(40);db(41);db(34);db(35);db(42);db(43);db(36);db(37);db(44);db(45);db(38);db(39);db(46);db(47);db(48);db(49);db(56);db(57);db(50);db(51);db(58);db(59);db(52);db(53);db(60);db(61);db(54);db(55);db(62);db(63);
    }
};

// bool cmpf(MKL_Complex8 a, Complex_int16 b, float epsilon = TOLERANCE) {
//     return (fabs(a.real - fixedToFloat(b.real)) < epsilon) && (fabs(a.imag - fixedToFloat(b.imag)) < epsilon);
// }

// bool vectorsEqual(MKL_Complex8* src, Complex_int16* mine, long m) {
//     for(int i = 0; i < m; i++) {
//         if(!(cmpf(src[i], mine[i])))
//             return false;
//     }
//     return true;
// }

// double runJITCGEMM(MKL_Complex8* a, MKL_Complex8* b, MKL_Complex8* c, MKL_INT m, MKL_INT k, int numIter) {
//     if(numIter == 0) return 0.0;
//     MKL_Complex8 alpha = {1, 0};
//     MKL_Complex8 beta = {0, 0};
//     MKL_INT lda = m;
//     MKL_INT ldb = k;
//     MKL_INT ldc = m;
//     double start = getTime();
//     // Create a handle and generate GEMM kernel
//     void* jitter;
//     mkl_jit_status_t status = mkl_jit_create_cgemm(&jitter, MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, m, 1, k, &alpha, lda, ldb, &beta, ldc);
//     if (MKL_JIT_ERROR == status) {
//         fprintf(stderr, "Error: insufficient memory to JIT and store the DGEMM kernel\n");
//         exit(1);
//     }
//     // Get kernel associated with handle
//     cgemm_jit_kernel_t my_cgemm = mkl_jit_get_cgemm_ptr(jitter);
//     for(int i = 0; i < numIter; i++)
//         my_cgemm(jitter, a, b, c); // Repeatedly execute the GEMM kernel
//     // Destroy the created handle/GEMM kernel
//     mkl_jit_destroy((void*)my_cgemm);
//     double ret = timeSince(start);
//     return ret;
// }



// enum PROGRAM_MODE {
//     MKL,
//     BYL,
//     BOTH
// };

// void benchDimensions(long m, long k, long numIter, PROGRAM_MODE mode, bool real) {
//     MKL_Complex8 *mat, *vec, *res;
//     Complex_int16 *mat16, *vec16, *res16;
//     // Allocate memory for matrix, vector, and resulting vector aligned on 64 byte boundary
//     mat = (MKL_Complex8*)mkl_calloc(m*k, sizeof(MKL_Complex8), 64);
//     vec = (MKL_Complex8*)mkl_calloc(k, sizeof(MKL_Complex8), 64);
//     res = (MKL_Complex8*)mkl_calloc(m, sizeof(MKL_Complex8), 64);
//     // Int16 version
//     mat16 = (Complex_int16*)aligned_alloc(128, m*k*sizeof(Complex_int16));
//     vec16 = (Complex_int16*)aligned_alloc(128, k*sizeof(Complex_int16));
//     res16 = (Complex_int16*)aligned_alloc(128, m*sizeof(Complex_int16));
//     memset(res16, 0, m*sizeof(Complex_int16));
//     // Randomly generate matrix/vector with values from 0 to 50
//     std::random_device rd;
//     std::mt19937 gen(10);
//     std::uniform_real_distribution<float> mat_dis(-1.0, 1.0);
//     int mod = 10; //rand()%mod-range
//     int range = mod/2;
//     for(int i = 0; i < m*k; i++) {
//         if(real) {
//             mat[i] = {mat_dis(gen), mat_dis(gen)};
//             mat16[i] = {floatToFixed(mat[i].real), floatToFixed(mat[i].imag)};
//         } else { // integer
//             mat16[i] = {(int16_t)(rand()%mod-range), (int16_t)(rand()%mod-range)};
//             mat[i] = {(float)mat16[i].real, (float)mat16[i].imag};
//         }
//         // std::cout << "(" << mat[i].real << "," << mat[i].imag << ")" << std::endl;
//         // std::cout << "(" << fixedToFloat(mat16[i].real) << "," << fixedToFloat(mat16[i].imag) << ")" << std::endl << std::endl;

//     }
//     std::uniform_real_distribution<float> vec_dis(-5.0, 5.0);
//     for(int i = 0; i < k; i++) {
//         if(real) {
//             vec[i] = {vec_dis(gen), vec_dis(gen)};
//             vec16[i] = {floatToFixed(vec[i].real), floatToFixed(vec[i].imag)};
//         } else {
//             vec16[i] = {(int16_t)(rand()%mod-range), (int16_t)(rand()%mod-range)};
//             vec[i] = {(float)vec16[i].real, (float)vec16[i].imag};
//         }
//         // std::cout << "(" << vec[i].real << "," << vec[i].imag << ")" << std::endl;
//         // std::cout << "(" << fixedToFloat(vec16[i].real) << "," << fixedToFloat(vec16[i].imag) << ")" << std::endl << std::endl;
//     }
//     double mklTime = 0.0;
//     double start = getTime();
//     // Uncomment below for int16
//     if(mode == BYL || mode == BOTH) {
//         JitInt16MatVec jit16(m, k);
//         jit16.setProtectModeRE(); // Use Read/Exec mode for security
//         void (*matvec16)(const Complex_int16*, const Complex_int16*, Complex_int16*) = jit16.getCode<void (*)(const Complex_int16*, const Complex_int16*, Complex_int16*)>();
//         for(int i = 0; i < numIter; i++)
//             matvec16(mat16, vec16, res16);
//     }
//     double myTime = timeSince(start);

//     // Uncomment below for MKL
//     if(mode == MKL || mode == BOTH) {
//         mklTime = runJITCGEMM(mat, vec, res, m, k, numIter);
//     }
//     // Save .asm of my function (MKL .asm is saved in runJITCGEMM)
//     // Output result
//     for(int i = 0; i < m; i++) std::cout << "(" << std::fixed << std::setprecision(2) << res[i].real << "," << std::fixed << std::setprecision(2) << res[i].imag << ")";
//     std::cout << std::endl;
//     for(int i = 0; i < m; i++) {
//         if(real)
//             std::cout << "(" << std::fixed << std::setprecision(2) << fixedToFloat(res16[i].real) << "," << std::fixed << std::setprecision(2) << fixedToFloat(res16[i].imag) << ")";
//         else
//             std::cout << res16[i];
//     }
//     std::cout << std::endl;
//     printf("\n        ---------- \n\n");
//     printf("     %ld iterations, (%ldx%ld) * (%ldx%d)\n", numIter, m, k, k, 1);
//     printf("MKL JIT cgemm: %.3f µs per iteration\n", mklTime/(double)numIter);
//     printf(" my JIT int16: %.3f µs per iteration\n", myTime/(double)numIter);
//     #define RESET   "\033[0m" // Terminal color codes
//     #define BOLDGREEN   "\033[1m\033[32m" 
//     std::cout << "  " << BOLDGREEN << std::fixed << std::setprecision(2) << mklTime/myTime << "x" << RESET << " MKL JIT cgemm" << std::endl;
//     std::cout << "---------------------------------\n" << std::endl;
//     // Assert resulting values are equal
//     if(mode == BOTH) assert(vectorsEqual(res, res16, m));
//     dimensions.push_back(std::to_string(m) + "x" + std::to_string(k));
//     mklTimes.push_back(mklTime/(double)numIter);
//     myTimes.push_back(myTime/(double)numIter);
//     // Free allocated memory
//     mkl_free(mat); mkl_free(vec); mkl_free(res);
//     free(mat16); free(vec16); free(res16);
// }


// int main(int argc, char** argv) {
//     srand(time(0));
//     long numIter = 1000000;
//     PROGRAM_MODE mode = BOTH;
//     char* nPtr = NULL;
//     long m = strtoul(argv[1], &nPtr, 0);
//     long k = strtoul(nPtr+1, &nPtr, 0);
//     if     (strcmp("mkl", argv[2]) == 0)  mode = MKL;
//     else if(strcmp("mine", argv[2]) == 0) mode = MINE;
//     else if(strcmp("both", argv[2]) == 0) mode = BOTH;
//     double exp = 6.0;
//     if(argc >= 4) {
//         exp = strtod(argv[3], NULL);
//     }
//     numIter = static_cast<long>(pow(10.0,exp));
//     bool real = true;
//     if(argc >= 5) {
//         real = false;
//     }
//     //for(long m = 16; m <= 208; m+=16) {
//         //for(long k = 16; k <= ; k += 16)
//             benchDimensions(m, k, numIter, mode, real);    
//     //}
//     //outputCSV("squares");
//     return 0;
// }

// Check power
// int main(int argc, char** argv) {
//     if(argc != 2) DIE("Requires only 1 argument, \"MKL\", \"BYL\" or \"BOTH\"\n");
//     long m = 64;
//     long k = 16;
//     long numIter = 1000000000;
//     if(strcmp(argv[1], "MKL") == 0) {
//         benchDimensions(m, k, numIter, MKL, true);
//     }
//     else if(strcmp(argv[1], "BYL") == 0) {
//         benchDimensions(m, k, numIter, BYL, true);
//     }
//     else if(strcmp(argv[1], "BOTH") == 0) {
//         benchDimensions(m, k, numIter, BOTH, true);
//     }
//     else {
//         DIE("Requires only 1 argument, \"MKL\", \"BYL\" or \"BOTH\"\n");
//     }
//     return 0;
// }
#endif