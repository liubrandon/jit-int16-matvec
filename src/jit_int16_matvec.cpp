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
#define DIE(...) fprintf(stderr, __VA_ARGS__); exit(1);
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
void outputASM(void* func, int m, int n, std::string filePrefix) {
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
    std::string filename = filePrefix + std::to_string(m) + "x" + std::to_string(n) + ".asm";
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
        : Xbyak::CodeGenerator(50*4096, Xbyak::DontSetProtectRWE) // Use Read/Exec mode for security
    {  // Input parameters rdi=&broad16, rsi=mat, rdx=vec, rcx=res, r8=&swapPairs (https://aaronbloomfield.github.io/pdr/book/x86-64bit-ccc-chapter.pdf)
        sub(rsp, 0x04*k); // allocate 256 bytes to the stack (size of 64 Complex_int16)
        vpbroadcastd(zmm31, dword [rdi]); //  rdi = {1, -1, ...}
        vmovdqu16(zmm0, zword [r8]);      // zmm0 = swapPairs

        for(int i = 0; i < k/16; i++) {
            // load vec into registers, swap pairs of vec, store swapped vector on the the stack
            vmovdqu16(zmm1, zword [rdx+(i*0x40)]);
            vpshufb(zmm1, zmm1, zmm0); 
            vmovdqu16(zword [rsp+(i*0x40)], zmm1); 
        }
        // first 16 elements of column 1
        vmovdqu16(zmm30, zword [rsi]); // load first column (a_b)
        vpbroadcastd(zmm5, dword [rdx]); // zmm5 = c_d (broadcast first two values from vec to all locations)
        vpmullw(zmm5, zmm5, zmm31); // zmm5 = c_minus_d (negate every other value in zmm5)
        vpbroadcastd(zmm6, dword [rsp]); /// zmm6 = d_c
        vpmaddwd(zmm29, zmm30, zmm5); // zmm29 = real_res accumulator
        vpmaddwd(zmm28, zmm30, zmm6); // zmm28 = imag_res accumulator
        if (m >= 32) {
            vmovdqu16(zmm30, zword [rsi+0x40]);
            vpmaddwd(zmm27, zmm30, zmm5);
            vpmaddwd(zmm26, zmm30, zmm6);
        }
        if (m >= 48) {
            vmovdqu16(zmm30, zword [rsi+0x80]);
            vpmaddwd(zmm25, zmm30, zmm5);
            vpmaddwd(zmm24, zmm30, zmm6);
        }
        if (m >= 64) {
            vmovdqu16(zmm30, zword [rsi+0xC0]);
            vpmaddwd(zmm23, zmm30, zmm5);
            vpmaddwd(zmm22, zmm30, zmm6);
        }        
        int rowsize = m*4;
        for(int i = 1; i < k; i++) {
            vmovdqu16(zmm30, zword [rsi+(rowsize*i)]);
            vpbroadcastd(zmm5, dword [rdx+(0x04*i)]);
            vpmullw(zmm5, zmm5, zmm31);
            vpbroadcastd(zmm6, dword [rsp+(0x04*i)]); 
            vpdpwssds(zmm29, zmm30, zmm5);
            vpdpwssds(zmm28, zmm30, zmm6);
            if (m >= 32) {
                vmovdqu16(zmm30, zword [rsi+(rowsize*i)+0x40]);
                vpdpwssds(zmm27, zmm30, zmm5);
                vpdpwssds(zmm26, zmm30, zmm6);
            } 
            if (m >= 48) {
                vmovdqu16(zmm30, zword [rsi+(rowsize*i)+0x80]);
                vpdpwssds(zmm25, zmm30, zmm5);
                vpdpwssds(zmm24, zmm30, zmm6);
            } 
            if (m >= 64) {
                vmovdqu16(zmm30, zword [rsi+(rowsize*i)+0xC0]);
                vpdpwssds(zmm23, zmm30, zmm5);
                vpdpwssds(zmm22, zmm30, zmm6);
            }
        }
        add(rsp, 0x04*k);

        vpslld(zmm28, zmm28, 0x10); // shift imag_res 16 bits left
        // Set up writemask k1
        mov(esi, 0xAAAAAAAA);
        kmovd(k1, esi);
        // Interleave real and imaginary
        vmovdqu16(zmm29 | k1, zmm28);
        // Write to memory
        vmovdqa64(zword [rcx], zmm29);
        if (m >= 32) {
            vpslld(zmm26, zmm26, 0x10); // shift imag_res 16 bits left
            vmovdqu16(zmm27 | k1, zmm26);
            vmovdqa64(zword [rcx+0x40], zmm27);
        }
        if (m >= 48) {
            vpslld(zmm24, zmm24, 0x10); // shift imag_res 16 bits left
            vmovdqu16(zmm25 | k1, zmm24);
            vmovdqa64(zword [rcx+0x80], zmm25);
        }
        if (m >= 64) {
            vpslld(zmm22, zmm22, 0x10); // shift imag_res 16 bits left
            vmovdqu16(zmm23 | k1, zmm22);
            vmovdqa64(zword [rcx+0xC0], zmm23);
        }
        ret();
    }
};

double runJITCGEMM(MKL_Complex8* a, MKL_Complex8* b, MKL_Complex8* c, MKL_INT m, MKL_INT k, int numIter) {
    MKL_Complex8 alpha = {1, 0};
    MKL_Complex8 beta = {0, 0};
    MKL_INT lda = m;
    MKL_INT ldb = k;
    MKL_INT ldc = m;
    void* jitter;
    double start = getTime();
    mkl_jit_status_t status = mkl_jit_create_cgemm(&jitter, MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, m, 1, k, &alpha, lda, ldb, &beta, ldc);
    // if (MKL_JIT_ERROR == status) {
    //     fprintf(stderr, "Error: insufficient memory to JIT and store the DGEMM kernel\n");
    //     exit(1);
    // }
    cgemm_jit_kernel_t my_cgemm = mkl_jit_get_cgemm_ptr(jitter);
    for(int i = 0; i < numIter; i++)
        my_cgemm(jitter, a, b, c); // execute the GEMM kernel
    double ret = timeSince(start);
    outputASM((void*)my_cgemm, m, k, "/asm/mkl");
    mkl_jit_destroy((void*)my_cgemm);
    return ret;
}

std::vector<std::string> dimensions;
std::vector<double> armaTimes, cgemvTimes, cgemmTimes, jitcgemmTimes, cgemvTimes_row, jitcgemmTimes_row;
void outputCSV() {
    std::ofstream outFile;
    outFile.open("results.csv");
    outFile << "Dimensions,Armadillo cgemv,cgemv,cgemv row major,cgemm,JIT cgemm,JIT cgemm row major\n";
    for(int i = 0; i < dimensions.size(); i++)
        outFile << dimensions[i] << "," << armaTimes[i] << "," << cgemvTimes[i] << "," << cgemvTimes_row[i] << "," << cgemmTimes[i] << "," << jitcgemmTimes[i] << "," << jitcgemmTimes_row[i] << ",\n";
    outFile.close();
}

int main(int argc, char** argv) {
    // Parse arguments and declare/initialize variables
    if(argc < 2) {
        DIE("Usage: %s MxK where M is # rows and K is # cols\n", argv[0]);
    }
    srand(time(0));
    long m, k, numIter;
    Complex_float *mat, *vec, *res, *res1, *res2;
    Complex_int16 *mat16, *vec16, *res16;
    char* nPtr;
    m = strtoul(argv[1], &nPtr, 0);
    k = strtoul(nPtr+1, NULL, 0);
    numIter = (argc >= 3) ? strtoul(argv[2], NULL, 0) : 1000000;
    // Allocate memory for matrix, vector, and resulting vector aligned on 64 byte boundary
    mat = (Complex_float*)aligned_alloc(64, m*k*sizeof(Complex_float));
    vec = (Complex_float*)aligned_alloc(64, k*sizeof(Complex_float));
    res = (Complex_float*)aligned_alloc(64, m*sizeof(Complex_float));
    memset(res, 0, m*sizeof(Complex_float));
    res1 = (Complex_float*)aligned_alloc(64, m*sizeof(Complex_float));
    memset(res1, 0, m*sizeof(Complex_float));
    res2 = (Complex_float*)aligned_alloc(64, m*sizeof(Complex_float));
    memset(res2, 0, m*sizeof(Complex_float));
    
    // Int16 version
    mat16 = (Complex_int16*)aligned_alloc(64, m*k*sizeof(Complex_int16));
    vec16 = (Complex_int16*)aligned_alloc(64, k*sizeof(Complex_int16));
    res16 = (Complex_int16*)aligned_alloc(64, m*sizeof(Complex_int16));
    memset(res16, 0, m*sizeof(Complex_int16));

    // Randomly generate matrix/vector with values from -range to range
    int mod = 10; //rand()%mod-range
    int range = mod/2;
    for(int i = 0; i < m*k; i++) {
        mat16[i] = {(int16_t)(rand()%mod-range), (int16_t)(rand()%mod-range)};
        mat[i] = {(float)mat16[i].real, (float)mat16[i].imag};
    }
    for(int i = 0; i < k; i++) {
        vec16[i] = {(int16_t)(rand()%mod-range), (int16_t)(rand()%mod-range)};
        vec[i] = {(float)vec16[i].real, (float)vec16[i].imag};
    }
    // for(int i = 0; i < k; i++) {
    //     for(int j = 0; j < m; j++)
    //         std::cout << mat16[i*m+j];
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // for(int i = 0; i < k; i++) std::cout << vec16[i];
    // std::cout << std::endl;

    double start = getTime();
    // JitFloatMatVec jit(m, k);
    // jit.setProtectModeRE(); // Use Read/Exec mode for security
    // void (*matvec)(void* notUsed, const Complex_float*, const Complex_float*, Complex_float*) = jit.getCode<void (*)(void* notUsed, const Complex_float*, const Complex_float*, Complex_float*)>();
    // for(int i = 0; i < numIter; i++) {
    //     matvec((void*)&broad, mat, vec, res);
    // }
    double myFloatTime = timeSince(start);
    start = getTime();
    // for(int i = 0; i < numIter; i++)
    //     matvecFloat_64x16((MKL_Complex8*)mat, (MKL_Complex8*)vec, (MKL_Complex8*)res2);
    double oldFloatTime = timeSince(start);
    start = getTime();
    JitInt16MatVec jit16(m, k);
    jit16.setProtectModeRE(); // Use Read/Exec mode for security
    void (*matvec16)(void* broad, const Complex_int16*, const Complex_int16*, Complex_int16*, void* swapPairs) = jit16.getCode<void (*)(void* broad, const Complex_int16*, const Complex_int16*, Complex_int16*, void* swapPairs)>();
    for(int i = 0; i < numIter; i++)
        matvec16((void*)&broad16, mat16, vec16, res16, (void*)&swapPairs);
    double myTime = timeSince(start);

    // Generate code at runtime (Just-in-Time) and output asm
    double mklTime = runJITCGEMM((MKL_Complex8*)mat, (MKL_Complex8*)vec, (MKL_Complex8*)res1, m, k, numIter);

    // Save .asm of each function
    //outputASM((void*)matvec, m, k, "./asm/myfloat");
    //outputASM((void*)matvec16, m, k, "./asm/myint16");

    // Output result
    // for(int i = 0; i < m; i++) std::cout << res[i];
    // std::cout << std::endl;
    for(int i = 0; i < m; i++) std::cout << res1[i];
    std::cout << std::endl;
    // for(int i = 0; i < m; i++) std::cout << res2[i];
    // std::cout << std::endl;
    for(int i = 0; i < m; i++) std::cout << res16[i];
    std::cout << std::endl;


    printf("\n        ---------- \n\n");
    printf("     %ld iterations, (%ldx%ld) * (%ldx%d)\n", numIter, m, k, k, 1);
    printf("MKL JIT cgemm: %.10f µs per iteration\n", mklTime/(double)numIter);
    // printf(" my JIT float: %.10f µs per iteration\n", myFloatTime/(double)numIter);
    // printf("    old float: %.10f µs per iteration\n", oldFloatTime/(double)numIter);
    printf(" my JIT int16: %.10f µs per iteration\n", myTime/(double)numIter);
    #define RESET   "\033[0m" // Terminal color codes
    #define BOLDGREEN   "\033[1m\033[32m" 
    std::cout << "  " << BOLDGREEN << std::fixed << std::setprecision(2) << mklTime/myTime << "x" << RESET << " MKL JIT cgemm" << std::endl;
    // // Free allocated memory
    // free(mat); free(vec); free(res); free(res1); free(res2);
    // free(mat16); free(vec16); free(res16);
}