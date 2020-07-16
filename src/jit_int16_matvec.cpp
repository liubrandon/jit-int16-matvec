#include <xbyak/xbyak.h>
#include <Zydis/Zydis.h>
#include <fstream>
#include <string.h>
#include <iostream>
#include "immintrin.h"
#include <complex>
#include "mkl.h"
#include "timer.hpp"
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
    const ZyanUSize length = 2000; // breaks on ret, should never reach length 7000
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

struct JitInt16MatVec : Xbyak::CodeGenerator {
    JitInt16MatVec(int m, int k)
        : Xbyak::CodeGenerator(4096, Xbyak::DontSetProtectRWE) // Use Read/Exec mode for security
    {  // Input parameters rdi=mat, rsi=vec, rdx=res (https://aaronbloomfield.github.io/pdr/book/x86-64bit-ccc-chapter.pdf)
        // vmovdqa64(zmm0, zword [rdi]);
        // vmovdqa64(zword [rdx], zmm0);
        for(int i = 0; i < m; i++) { // examples for copying data from matrix to res
            mov(r10d, dword [rdi+(i*4)]);
            mov(dword [rdx+(i*4)], r10d);
        }
        ret();
    }
};

Complex_float broad = {1.0,-1.0};
struct JitFloatMatVec : Xbyak::CodeGenerator {
    JitFloatMatVec(int m, int k)
        : Xbyak::CodeGenerator(4096, Xbyak::DontSetProtectRWE) // Use Read/Exec mode for security
    {  // Input parameters rdi=?, rsi=mat, rdx=vec, rcx=res
        vpbroadcastq(zmm31, ptr [rdi]);
        vmovups(zmm30, zword [rsi]);
        vmovups(zmm29, zword [rsi+0x40]);
        vmulps(zmm28, zmm30, ptr_b [rdx]);
        vmulps(zmm23, zmm30, ptr_b [rdx+0x04]);
        vmovups(zmm30, zword [rsi+0x80]);
        vmulps(zmm27, zmm29, ptr_b [rdx+0x08]);
        vmulps(zmm22, zmm29, ptr_b [rdx+0x0C]);
        vmovups(zmm29, zword [rsi+0xC0]);
        vmulps(zmm26, zmm30, ptr_b [rdx+0x10]);
        vmulps(zmm21, zmm30, ptr_b [rdx+0x14]);
        vmovups(zmm30, zword [rsi+0x100]);
        vmulps(zmm25, zmm29, ptr_b [rdx+0x18]);
        vmulps(zmm20, zmm29, ptr_b [rdx+0x1C]);
        vmovups(zmm29, zword [rsi+0x140]);
        vmulps(zmm24, zmm30, ptr_b [rdx+0x20]);
        vmulps(zmm19, zmm30, ptr_b [rdx+0x24]);
        vmovups(zmm30, zword [rsi+0x180]);
        vfmadd231ps(zmm28, zmm29, ptr_b [rdx+0x28]);
        vfmadd231ps(zmm23, zmm29, ptr_b [rdx+0x2C]);
        vmovups(zmm29, zword [rsi+0x1C0]);
        vfmadd231ps(zmm27, zmm30, ptr_b [rdx+0x30]);
        vfmadd231ps(zmm22, zmm30, ptr_b [rdx+0x34]);
        vfmadd231ps(zmm26, zmm29, ptr_b [rdx+0x38]);
        vfmadd231ps(zmm21, zmm29, ptr_b [rdx+0x3C]);
        vaddps(zmm28, zmm28, zmm24);
        vaddps(zmm23, zmm23, zmm19);
        vaddps(zmm28, zmm28, zmm26);
        vaddps(zmm23, zmm23, zmm21);
        vaddps(zmm27, zmm27, zmm25);
        vaddps(zmm22, zmm22, zmm20);
        vaddps(zmm28, zmm28, zmm27);
        vaddps(zmm23, zmm23, zmm22);
        vpermilps(zmm23, zmm23, 0xB1);
        vfnmadd231ps(zmm28, zmm23, zmm31);
        vmovups(zword [rcx], zmm28);
        vzeroupper();
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
    outputASM((void*)my_cgemm, m, k, "mkl");
    mkl_jit_destroy((void*)my_cgemm);
    return ret;
}

int main(int argc, char** argv) {
    // Parse arguments and declare/initialize variables
    if(argc != 2) {
        DIE("Usage: %s MxK where M is # rows and K is # cols\n", argv[0]);
    }
    srand(time(0));
    long numIter = 100000;
    int m, k;
    Complex_float *mat, *vec, *res, *res1;
    char* nPtr;
    m = strtoul(argv[1], &nPtr, 0);
    k = strtoul(nPtr+1, NULL, 0);

    // Allocate memory for matrix, vector, and resulting vector aligned on 64 byte boundary
    mat = (Complex_float*)aligned_alloc(64, m*k*sizeof(Complex_float));
    vec = (Complex_float*)aligned_alloc(64, k*sizeof(Complex_float));
    res = (Complex_float*)aligned_alloc(64, m*sizeof(Complex_float));
    memset(res, 0, m*sizeof(Complex_float));
    res1 = (Complex_float*)aligned_alloc(64, m*sizeof(Complex_float));
    memset(res1, 0, m*sizeof(Complex_float));

    // Randomly generate matrix/vector with values from -range to range
    int mod = 10; //rand()%mod-range
    int range = mod/2;
    for(int i = 0; i < m*k; i++) mat[i] = {(float)(rand()%mod-range), (float)(rand()%mod-range)};
    for(int i = 0; i < k; i++)   vec[i] = {(float)(rand()%mod-range), (float)(rand()%mod-range)};

    // Generate code at runtime (Just-in-Time) and output asm
    double start = getTime();
    JitFloatMatVec jit(m, k);
    jit.setProtectModeRE(); // Use Read/Exec mode for security
    void (*matvec)(void* notUsed, const Complex_float*, const Complex_float*, Complex_float*) = jit.getCode<void (*)(void* notUsed, const Complex_float*, const Complex_float*, Complex_float*)>();
    for(int i = 0; i < numIter; i++)
        matvec((void*)&broad, mat, vec, res);
    double myTime = timeSince(start);
    double mklTime = runJITCGEMM((MKL_Complex8*)mat, (MKL_Complex8*)vec, (MKL_Complex8*)res1, m, k, numIter);
    // Output result
    for(int i = 0; i < m; i++) std::cout << res[i];
    std::cout << std::endl;
    for(int i = 0; i < m; i++) std::cout << res1[i];
    std::cout << std::endl;
    outputASM((void*)matvec, m, k, "myfloat");

    printf("\n        ---------- \n\n");
    printf("     %ld iterations, (%dx%d) * (%dx%d)\n", numIter, m, k, k, 1);
    printf("MKL JIT cgemm: %.10f µs per iteration\n", mklTime/(double)numIter);
    printf(" my JIT float: %.10f µs per iteration\n", myTime/(double)numIter);
    // Free allocated memory
    free(mat); free(vec); free(res);
}