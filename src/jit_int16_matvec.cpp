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
#define DIE(...) fprintf(stderr, __VA_ARGS__); exit(1);
#define FIXED_POINT_FRACTIONAL_BITS 8

inline int16_t floatToFixed(float flo) {
    return (int16_t)(round(flo * (1 << FIXED_POINT_FRACTIONAL_BITS)));
}

inline float fixedToFloat(int16_t fix) {
    return ((float)fix / (float)(1 << FIXED_POINT_FRACTIONAL_BITS));
}

struct Complex_float {
    float real;
    float imag;
    friend std::ostream& operator<<(std::ostream& os, const Complex_float& c) {
        os << "(" << c.real << "," << c.imag << ")";
        return os;
    }
};
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
void outputASM(void* func, int m, int n, std::string filePrefix, std::string type) {
    // Initialize decoder context
    ZydisDecoder decoder;
    ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_ADDRESS_WIDTH_64);

    // Initialize formatter. Only required when you actually plan to do instruction
    // formatting ("disassembling"), like we do here
    ZydisFormatter formatter;
    ZydisFormatterInit(&formatter, ZYDIS_FORMATTER_STYLE_INTEL);

    // Loop over the instructions in our buffer.
    // The runtime-address (instruction pointer) is chosen arbitrary here in order to better
    // visualize relative addressing
    ZyanU64 runtime_address = (*(uintptr_t*)func);
    ZyanUSize offset = 0;
    const ZyanUSize length = 7000; // breaks on ret, should never reach length 7000
    ZydisDecodedInstruction instruction;

    // File to output ASM
    std::ofstream outFile;
    std::string filename = filePrefix + std::to_string(m) + "x" + std::to_string(n) + type + ".asm";
    outFile.open(filename);
    while (ZYAN_SUCCESS(ZydisDecoderDecodeBuffer(&decoder, (void*)((uintptr_t)func + offset), length - offset, &instruction))) {
        // Print current instruction pointer.
        // std::cout << "0x" << std::uppercase << std::hex << runtime_address << "   ";
        outFile << "0x" << std::uppercase << std::hex << runtime_address << "   ";
        // Format & print the binary instruction structure to human readable format
        char buffer[256];
        ZydisFormatterFormatInstruction(&formatter, &instruction, buffer, sizeof(buffer),
            runtime_address);
        //puts(buffer);
        outFile << buffer << "\n";
        offset += instruction.length;
        runtime_address += instruction.length;
        if(strstr(buffer, "ret") != NULL)
        break;
    }
    outFile.close();
}

Complex_int16 broad16 = {1, -1};
int8_t temp[64] = {2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13,18,19,16,17,22,23,20,21,26,27,24,25,30,31,28,29,34,35,
                                32,33,38,39,36,37,42,43,40,41,46,47,44,45,50,51,48,49,54,55,52,53,58,59,56,57,62,63,60,61};
int16_t temp1[32] = {-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1};
__m512i swapPairs = _mm512_loadu_si512((const void*)temp);
__m512i subAdd = _mm512_loadu_si512((const void*)temp1);

struct JitInt16MatVec : Xbyak::CodeGenerator {
    JitInt16MatVec(int m, int k)
        : Xbyak::CodeGenerator(200*4096, Xbyak::DontSetProtectRWE) // Use Read/Exec mode for security
    {  // Input parameters rdi=&broad16, rsi=mat, rdx=vec, rcx=res, r8=&swapPairs
        int rowsize = m*4;
        sub(rsp, 0x04*k); // allocate vec size onto stack
        vpbroadcastd(zmm31, dword [rdi]); //  rdi = {1, -1, ...}
        vmovdqu16(zmm0, zword [r8]);      // zmm0 = swapPairs
        // Swap pairs of vector and store that copy on the stack
        for(int i = 0; i < k/16; i++) {
            vmovdqu16(zmm1, zword [rdx+(i*0x40)]);
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
                vmovdqu16(zmm30, zword [rsi+(rowsize*i)]);
                vmovdqu16(zmm27, zword [rsi+(rowsize*(i+1))]);
                vmovdqu16(zmm26, zword [rsi+(rowsize*(i+2))]);
                vmovdqu16(zmm25, zword [rsi+(rowsize*(i+3))]);

                vpbroadcastd( zmm5, dword [rdx+(0x04*i)]);
                vpmullw(zmm5 , zmm5 , zmm31);
                vpbroadcastd( zmm6, dword [rsp+(0x04*i)]);
                vpmaddwd(zmm5, zmm30, zmm5);
                vpmaddwd(zmm6, zmm30, zmm6);
                vpsllw(zmm5, zmm5, FIXED_POINT_FRACTIONAL_BITS);
                vpsllw(zmm6, zmm6, FIXED_POINT_FRACTIONAL_BITS);
                vpaddd(zmm29, zmm29, zmm5);
                vpaddd(zmm28, zmm29, zmm6);

                vpbroadcastd( zmm7, dword [rdx+(0x04*(i+1))]);
                vpmullw(zmm7 , zmm7 , zmm31);
                vpbroadcastd( zmm8, dword [rsp+(0x04*(i+1))]);
                vpmaddwd(zmm7, zmm27, zmm7);
                vpmaddwd(zmm8, zmm27, zmm8);
                vpsllw(zmm7, zmm7, FIXED_POINT_FRACTIONAL_BITS);
                vpsllw(zmm8, zmm8, FIXED_POINT_FRACTIONAL_BITS);
                vpaddd(zmm1, zmm27, zmm7);
                vpaddd(zmm2, zmm27, zmm8);

                vpbroadcastd( zmm9, dword [rdx+(0x04*(i+2))]);
                vpmullw(zmm9 , zmm9 , zmm31);
                vpbroadcastd(zmm10, dword [rsp+(0x04*(i+2))]);
                vpmaddwd(zmm9, zmm26, zmm9);
                vpmaddwd(zmm10, zmm26, zmm10);
                vpsllw(zmm9, zmm9, FIXED_POINT_FRACTIONAL_BITS);
                vpsllw(zmm10, zmm10, FIXED_POINT_FRACTIONAL_BITS);
                vpaddd(zmm3, zmm26, zmm9);
                vpaddd(zmm4, zmm26, zmm10);

                vpbroadcastd(zmm11, dword [rdx+(0x04*(i+3))]);
                vpmullw(zmm11, zmm11, zmm31);
                vpbroadcastd(zmm12, dword [rsp+(0x04*(i+3))]); 
                vpmaddwd(zmm11, zmm25, zmm11);
                vpmaddwd(zmm12, zmm25, zmm12);
                vpsllw(zmm11, zmm11, FIXED_POINT_FRACTIONAL_BITS);
                vpsllw(zmm12, zmm12, FIXED_POINT_FRACTIONAL_BITS);
                vpaddd(zmm21, zmm25, zmm11);
                vpaddd(zmm22, zmm25, zmm12);
            }
            // Sum up accumulators for imaginary results
            vpaddd(zmm22,zmm22,zmm2);
            vpaddd(zmm28,zmm28,zmm4);
            vpaddd(zmm28,zmm28,zmm22);
            // Sum up accumulators for real results
            vpaddd(zmm29,zmm29,zmm3);
            vpaddd(zmm21,zmm21,zmm1);
            vpaddd(zmm29,zmm29,zmm21);

            vpslld(zmm28, zmm28, 0x10); // shift imag_res 16 bits left
            // Set up writemask k1
            mov(esi, 0xAAAAAAAA);
            kmovd(k1, esi);
            
            // Interleave real and imaginary
            vmovdqu16(zmm29 | k1, zmm28);
            // Write everything to memory 
            vmovdqa64(zword [rcx], zmm29);
        }
        // Other sizes take this kernel
        else {
            int inc = 0; // overwritten in the loop
            for(int r = 0; r < m; r += inc) { // TODO: Fix this loop
                // first 16 elements of column 1
                vmovdqu16(zmm30, zword [rsi+(r*0x04)]); // load first column (a_b)
                vpbroadcastd(zmm5, dword [rdx]); // zmm5 = c_d (broadcast first two values from vec to all locations)
                vpmullw(zmm5, zmm5, zmm31); // zmm5 = c_minus_d (negate every other value in zmm5)
                vpbroadcastd(zmm6, dword [rsp]); // zmm6 = d_c
                vpmaddwd(zmm28, zmm30, zmm6); // zmm28 = imag_res accumulator
                vpmaddwd(zmm29, zmm30, zmm5); // zmm29 = real_res accumulator
                inc = 16;
                if (m-r >= 32) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+0x40]);
                    vpmaddwd(zmm27, zmm30, zmm5);
                    vpmaddwd(zmm26, zmm30, zmm6); inc = 32;
                }
                if (m-r >= 48) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+0x80]);
                    vpmaddwd(zmm25, zmm30, zmm5);
                    vpmaddwd(zmm24, zmm30, zmm6); inc = 48;
                }
                if (m-r >= 64) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+0xC0]);
                    vpmaddwd(zmm23, zmm30, zmm5);
                    vpmaddwd(zmm22, zmm30, zmm6); inc = 64;
                }        
                if (m-r >= 80) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+0x100]);
                    vpmaddwd(zmm21, zmm30, zmm5);
                    vpmaddwd(zmm20, zmm30, zmm6); inc = 80;
                }
                if (m-r >= 96) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+0x140]);
                    vpmaddwd(zmm19, zmm30, zmm5);
                    vpmaddwd(zmm18, zmm30, zmm6); inc = 96;
                }
                if (m-r >= 112) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+0x180]);
                    vpmaddwd(zmm17, zmm30, zmm5);
                    vpmaddwd(zmm16, zmm30, zmm6); inc = 112;
                }        
                if (m-r >= 128) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+0x1C0]);
                    vpmaddwd(zmm15, zmm30, zmm5);
                    vpmaddwd(zmm14, zmm30, zmm6); inc = 128;
                }        
                if (m-r >= 144) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+0x200]);
                    vpmaddwd(zmm13, zmm30, zmm5);
                    vpmaddwd(zmm12, zmm30, zmm6); inc = 144;
                }
                if (m-r >= 160) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+0x240]);
                    vpmaddwd(zmm11, zmm30, zmm5);
                    vpmaddwd(zmm10, zmm30, zmm6); inc = 160;
                }
                if (m-r >= 176) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+0x280]);
                    vpmaddwd(zmm9 , zmm30, zmm5);
                    vpmaddwd(zmm8 , zmm30, zmm6); inc = 176;
                }        
                if (m-r >= 192) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+0x2C0]);
                    vpmaddwd(zmm7 , zmm30, zmm5);
                    vpmaddwd(zmm4 , zmm30, zmm6); inc = 192;
                }        
                if (m-r >= 208) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+0x300]);
                    vpmaddwd(zmm3 , zmm30, zmm5);
                    vpmaddwd(zmm2 , zmm30, zmm6); inc = 208;
                } 
                for(int i = 1; i < k; i++) {
                    vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)]);
                    vpbroadcastd(zmm5, dword [rdx+(0x04*i)]);
                    vpmullw(zmm5, zmm5, zmm31);
                    vpbroadcastd(zmm6, dword [rsp+(0x04*i)]); 
                    vpdpwssds(zmm29, zmm30, zmm5);
                    vpdpwssds(zmm28, zmm30, zmm6);
                    if (m-r >= 32) {
                        vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)+0x40]);
                        vpdpwssds(zmm27, zmm30, zmm5);
                        vpdpwssds(zmm26, zmm30, zmm6);
                    } 
                    if (m-r >= 48) {
                        vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)+0x80]);
                        vpdpwssds(zmm25, zmm30, zmm5);
                        vpdpwssds(zmm24, zmm30, zmm6);
                    } 
                    if (m-r >= 64) {
                        vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)+0xC0]);
                        vpdpwssds(zmm23, zmm30, zmm5);
                        vpdpwssds(zmm22, zmm30, zmm6);
                    }
                    if (m-r >= 80) {
                        vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)+0x100]);
                        vpdpwssds(zmm21, zmm30, zmm5);
                        vpdpwssds(zmm20, zmm30, zmm6);
                    } 
                    if (m-r >= 96) {
                        vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)+0x140]);
                        vpdpwssds(zmm19, zmm30, zmm5);
                        vpdpwssds(zmm18, zmm30, zmm6);
                    } 
                    if (m-r >= 112) {
                        vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)+0x180]);
                        vpdpwssds(zmm17, zmm30, zmm5);
                        vpdpwssds(zmm16, zmm30, zmm6);
                    }
                    if (m-r >= 128) {
                        vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)+0x1C0]);
                        vpdpwssds(zmm15, zmm30, zmm5);
                        vpdpwssds(zmm14, zmm30, zmm6);
                    }
                    if (m-r >= 144) {
                        vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)+0x200]);
                        vpdpwssds(zmm13, zmm30, zmm5);
                        vpdpwssds(zmm12, zmm30, zmm6);
                    } 
                    if (m-r >= 160) {
                        vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)+0x240]);
                        vpdpwssds(zmm11, zmm30, zmm5);
                        vpdpwssds(zmm10, zmm30, zmm6);
                    } 
                    if (m-r >= 176) {
                        vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)+0x280]);
                        vpdpwssds(zmm9 , zmm30, zmm5);
                        vpdpwssds(zmm8 , zmm30, zmm6);
                    }
                    if (m-r >= 192) {
                        vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)+0x2C0]);
                        vpdpwssds(zmm7 , zmm30, zmm5);
                        vpdpwssds(zmm4 , zmm30, zmm6);
                    }
                    if (m-r >= 208) {
                        vmovdqu16(zmm30, zword [rsi+(r*0x04)+(rowsize*i)+0x300]);
                        vpdpwssds(zmm3 , zmm30, zmm5);
                        vpdpwssds(zmm2 , zmm30, zmm6);
                    }
                }
                vpslld(zmm28, zmm28, 0x10); // shift imag_res 16 bits left
                // Set up writemask k1
                mov(esi, 0xAAAAAAAA);
                kmovd(k1, esi);
                // Interleave real and imaginary
                vmovdqu16(zmm29 | k1, zmm28);
                // Write to memory
                vmovdqa64(zword [rcx+(r*0x04)], zmm29);
                if (m-r >= 32) {
                    vpslld(zmm26, zmm26, 0x10);
                    vmovdqu16(zmm27 | k1, zmm26);
                    vmovdqa64(zword [rcx+(r*0x04)+0x40], zmm27);
                }
                if (m-r >= 48) {
                    vpslld(zmm24, zmm24, 0x10);
                    vmovdqu16(zmm25 | k1, zmm24);
                    vmovdqa64(zword [rcx+(r*0x04)+0x80], zmm25);
                }
                if (m-r >= 64) {
                    vpslld(zmm22, zmm22, 0x10);
                    vmovdqu16(zmm23 | k1, zmm22);
                    vmovdqa64(zword [rcx+(r*0x04)+0xC0], zmm23);
                }
                if (m-r >= 80) {
                    vpslld(zmm20, zmm20, 0x10);
                    vmovdqu16(zmm21 | k1, zmm20);
                    vmovdqa64(zword [rcx+(r*0x04)+0x100], zmm21);
                }
                if (m-r >= 96) {
                    vpslld(zmm18, zmm18, 0x10);
                    vmovdqu16(zmm19 | k1, zmm18);
                    vmovdqa64(zword [rcx+(r*0x04)+0x140], zmm19);
                }
                if (m-r >= 112) {
                    vpslld(zmm16, zmm16, 0x10);
                    vmovdqu16(zmm17 | k1, zmm16);
                    vmovdqa64(zword [rcx+(r*0x04)+0x180], zmm17);
                }
                if (m-r >= 128) {
                    vpslld(zmm14, zmm14, 0x10);
                    vmovdqu16(zmm15 | k1, zmm14);
                    vmovdqa64(zword [rcx+(r*0x04)+0x1C0], zmm15);
                }
                if (m-r >= 144) {
                    vpslld(zmm12, zmm12, 0x10);
                    vmovdqu16(zmm13 | k1, zmm12);
                    vmovdqa64(zword [rcx+(r*0x04)+0x200], zmm13);
                }
                if (m-r >= 160) {
                    vpslld(zmm10, zmm10, 0x10);
                    vmovdqu16(zmm11 | k1, zmm10);
                    vmovdqa64(zword [rcx+(r*0x04)+0x240], zmm11);
                }
                if (m-r >= 176) {
                    vpslld(zmm8, zmm8, 0x10);
                    vmovdqu16(zmm9 | k1, zmm8);
                    vmovdqa64(zword [rcx+(r*0x04)+0x280], zmm9);
                }
                if (m-r >= 192) {
                    vpslld(zmm4, zmm4, 0x10);
                    vmovdqu16(zmm7 | k1, zmm4);
                    vmovdqa64(zword [rcx+(r*0x04)+0x2C0], zmm7);
                }
                if (m-r >= 208) {
                    vpslld(zmm2, zmm2, 0x10);
                    vmovdqu16(zmm3 | k1, zmm2);
                    vmovdqa64(zword [rcx+(r*0x04)+0x300], zmm3);
                }
            }
        }
        add(rsp, 0x04*k); // deallocate vec size from stack
        vzeroupper(); // needed? I think needed if using both AVX2 and other SIMD extensions
        ret();
    }
};

double runJITCGEMM(MKL_Complex8* a, MKL_Complex8* b, MKL_Complex8* c, MKL_INT m, MKL_INT k, int numIter) {
    if(numIter == 0) return 0.0;
    MKL_Complex8 alpha = {1, 0};
    MKL_Complex8 beta = {0, 0};
    MKL_INT lda = m;
    MKL_INT ldb = k;
    MKL_INT ldc = m;
    double start = getTime();
    // Create a handle and generate GEMM kernel
    void* jitter;
    mkl_jit_status_t status = mkl_jit_create_cgemm(&jitter, MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, m, 1, k, &alpha, lda, ldb, &beta, ldc);
    if (MKL_JIT_ERROR == status) {
        fprintf(stderr, "Error: insufficient memory to JIT and store the DGEMM kernel\n");
        exit(1);
    }
    // Get kernel associated with handle
    cgemm_jit_kernel_t my_cgemm = mkl_jit_get_cgemm_ptr(jitter);
    for(int i = 0; i < numIter; i++)
        my_cgemm(jitter, a, b, c); // Repeatedly execute the GEMM kernel
    // Destroy the created handle/GEMM kernel
    mkl_jit_destroy((void*)my_cgemm);
    double ret = timeSince(start);
    outputASM((void*)my_cgemm, m, k, std::string("./asm/") + std::to_string(m) + "xK/", "_mkl");
    return ret;
}

std::vector<std::string> dimensions;
std::vector<double> mklTimes, myTimes;
void outputCSV(std::string filename) {
    std::ofstream outFile;
    outFile.open(filename+".csv");
    outFile << "Dimensions,MKL,My int16,Speedup,";
    outFile << "Overall avg. speedup:," << std::accumulate(mklTimes.begin(), mklTimes.end(), 0.0)/(double)std::accumulate(myTimes.begin(), myTimes.end(), 0.0) << ",\n";
    for(int i = 0; i < dimensions.size(); i++)
        outFile << dimensions[i] << "," << mklTimes[i] << "," << myTimes[i] << "," << mklTimes[i]/myTimes[i] << ",\n";
    outFile.close();
    dimensions.clear(); mklTimes.clear(); myTimes.clear();
}

bool vectorsEqual(float* vec1, int16_t* vec2, long size) {
    for(int i = 0; i < size*2; i++) {
        if((int16_t)vec1[i] != vec2[i])
            return false;
    }
    return true;
}
enum PROGRAM_MODE {
    MKL,
    MINE,
    BOTH
};
void benchDimensions(long m, long k, long numIter, PROGRAM_MODE mode) {
    MKL_Complex8 *mat, *vec, *res;
    Complex_int16 *mat16, *vec16, *res16;
    // Allocate memory for matrix, vector, and resulting vector aligned on 64 byte boundary
    mat = (MKL_Complex8*)mkl_calloc(m*k, sizeof(MKL_Complex8), 64);
    vec = (MKL_Complex8*)mkl_calloc(k, sizeof(MKL_Complex8), 64);
    res = (MKL_Complex8*)mkl_calloc(m, sizeof(MKL_Complex8), 64);
    // Int16 version
    mat16 = (Complex_int16*)aligned_alloc(128, m*k*sizeof(Complex_int16));
    vec16 = (Complex_int16*)aligned_alloc(128, k*sizeof(Complex_int16));
    res16 = (Complex_int16*)aligned_alloc(128, m*sizeof(Complex_int16));
    memset(res16, 0, m*sizeof(Complex_int16));
    // Randomly generate matrix/vector with values from 0 to 50
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-15.0, 15.0);
    for(int i = 0; i < m*k; i++) {
        mat[i] = {dis(gen), dis(gen)};
        mat16[i] = {floatToFixed(mat[i].real), floatToFixed(mat[i].imag)};
        // std::cout << "(" << mat[i].real << "," << mat[i].imag << ")" << std::endl;
        // std::cout << "(" << fixedToFloat(mat16[i].real) << "," << fixedToFloat(mat16[i].imag) << ")" << std::endl << std::endl;

    }
    for(int i = 0; i < k; i++) {
        vec[i] = {dis(gen), dis(gen)};
        vec16[i] = {floatToFixed(vec[i].real), floatToFixed(vec[i].imag)};
        // std::cout << "(" << vec[i].real << "," << vec[i].imag << ")" << std::endl;
        // std::cout << "(" << fixedToFloat(vec16[i].real) << "," << fixedToFloat(vec16[i].imag) << ")" << std::endl << std::endl;
    }
    double mklTime = 0.0;
    double start = getTime();
    // Uncomment below for int16
    if(mode == MINE || mode == BOTH) {
        JitInt16MatVec jit16(m, k);
        jit16.setProtectModeRE(); // Use Read/Exec mode for security
        void (*matvec16)(void* broad, const Complex_int16*, const Complex_int16*, Complex_int16*, void* swapPairs) = jit16.getCode<void (*)(void* broad, const Complex_int16*, const Complex_int16*, Complex_int16*, void* swapPairs)>();
        for(int i = 0; i < numIter; i++)
            matvec16((void*)&broad16, mat16, vec16, res16, (void*)&swapPairs);
        //outputASM((void*)matvec16, m, k, std::string("./asm/") + std::to_string(m) + std::string("xK/"), "_myint16");
    }
    double myTime = timeSince(start);

    // Uncomment below for MKL
    if(mode == MKL || mode == BOTH) {
        mklTime = runJITCGEMM(mat, vec, res, m, k, numIter);
    }
    // Save .asm of my function (MKL .asm is saved in runJITCGEMM)
    // Output result
    for(int i = 0; i < m; i++) std::cout << "(" << res[i].real << "," << res[i].imag << ")";
    std::cout << std::endl;
    for(int i = 0; i < m; i++) std::cout << "(" << fixedToFloat(res16[i].real) << "," << fixedToFloat(res16[i].imag) << ")";
    std::cout << std::endl;
    printf("\n        ---------- \n\n");
    printf("     %ld iterations, (%ldx%ld) * (%ldx%d)\n", numIter, m, k, k, 1);
    printf("MKL JIT cgemm: %.3f µs per iteration\n", mklTime/(double)numIter);
    printf(" my JIT int16: %.3f µs per iteration\n", myTime/(double)numIter);
    #define RESET   "\033[0m" // Terminal color codes
    #define BOLDGREEN   "\033[1m\033[32m" 
    std::cout << "  " << BOLDGREEN << std::fixed << std::setprecision(2) << mklTime/myTime << "x" << RESET << " MKL JIT cgemm" << std::endl;
    std::cout << "---------------------------------\n" << std::endl;
    // Assert resulting values are equal
    if(mode == BOTH) assert(vectorsEqual((float*)res, (int16_t*)res16, m));
    dimensions.push_back(std::to_string(m) + "x" + std::to_string(k));
    mklTimes.push_back(mklTime/(double)numIter);
    myTimes.push_back(myTime/(double)numIter);
    // Free allocated memory
    mkl_free(mat); mkl_free(vec); mkl_free(res);
    free(mat16); free(vec16); free(res16);
}

int main(int argc, char** argv) {
    float pi =  2 * std::acos(0.0);
    std::cout << pi*2 << std::endl;
    int16_t pi_fixed = floatToFixed(pi);
    int16_t two = floatToFixed((float)2.0);
    two = two >> 8;
    pi_fixed *= two;
    float new_pi = fixedToFloat(pi_fixed);
    std::cout << new_pi << std::endl;
    return 0;
    srand(time(0));
    long numIter = 1000000;
    PROGRAM_MODE mode = BOTH;
    char* nPtr = NULL;
    long m = strtoul(argv[1], &nPtr, 0);
    long k = strtoul(nPtr+1, &nPtr, 0);
    if     (strcmp("mkl", argv[2]) == 0)  mode = MKL;
    else if(strcmp("mine", argv[2]) == 0) mode = MINE;
    else if(strcmp("both", argv[2]) == 0) mode = BOTH;
    double exp = 6.0;
    if(argc >= 4) {
        exp = strtod(argv[3], NULL);
    }
    numIter = static_cast<long>(pow(10.0,exp));
    //for(long m = 16; m <= 208; m+=16) {
        //for(long k = 16; k <= ; k += 16)
            benchDimensions(m, k, numIter, mode);    
    //}
    //outputCSV("squares");
    return 0;
}