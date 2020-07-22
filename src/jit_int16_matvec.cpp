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
#define DIE(...) fprintf(stderr, __VA_ARGS__); exit(1);
#define my_vfmadd231w(accu, one, two) ({ \
    vpmullw(one, one, two); \
    vpaddw(accu, accu, one); \
})
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
        : Xbyak::CodeGenerator(4096, Xbyak::DontSetProtectRWE) // Use Read/Exec mode for security
    {  // Input parameters rdi=&broad16, rsi=mat, rdx=vec, rcx=res, r8=&swapPairs (https://aaronbloomfield.github.io/pdr/book/x86-64bit-ccc-chapter.pdf)
        sub(rsp, 0x100); // allocate 256 bytes to the stack (size of 64 Complex_int16)
        vpbroadcastd(zmm31, dword [rdi]); //  rdi = {1, -1, ...}
        vmovdqu16(zmm0, zword [r8]);      // zmm0 = swapPairs
        vmovdqu16(zmm30, zword [rsi]); // load first column (a_b)
        vmovdqu16(zmm1, zword [rdx]);
        vmovdqu16(zmm2, zword [rdx+0x40]);
        vmovdqu16(zmm3, zword [rdx+0x80]);
        vmovdqu16(zmm4, zword [rdx+0xC0]);
        vpshufb(zmm1, zmm1, zmm0); // swap pairs of vector
        vpshufb(zmm2, zmm2, zmm0);
        vpshufb(zmm3, zmm3, zmm0);
        vpshufb(zmm4, zmm4, zmm0);
        vmovdqu16(zword [rsp],zmm1); // store swapped vector into the stack
        vmovdqu16(zword [rsp+0x40],zmm2);
        vmovdqu16(zword [rsp+0x80],zmm3);
        vmovdqu16(zword [rsp+0xC0],zmm4);
        // first iteration use madd
        vpbroadcastd(zmm5, dword [rdx]); // zmm5 = c_d (broadcast first two values from vec to all locations)
        vpmullw(zmm5, zmm5, zmm31); // zmm5 = c_minus_d (negate every other value in zmm5)
        vpmaddwd(zmm29, zmm30, zmm5); // zmm29 = real_res accumulator
        vpbroadcastd(zmm6, dword [rsp]); /// zmm6 = d_c
        vpmaddwd(zmm28, zmm30, zmm6); // zmm28 = imag_res accumulator
        for(int i = 1; i < 64; i++) {
            add(rsi, 0x40); // advance to next column of mat
            add(rdx, 0x04); // advance 4 bytes to next complex number of vec
            add(rsp, 0x04); // advance 4 bytes to next complex number of vec_swapped
            vmovdqu16(zmm30, zword [rsi]);
            vpbroadcastd(zmm5, dword [rdx]);
            vpmullw(zmm5, zmm5, zmm31);
            vpdpwssds(zmm29, zmm30, zmm5);
            vpbroadcastd(zmm6, dword [rsp]); 
            vpdpwssds(zmm28, zmm30, zmm6);
        }
        add(rsp, 0x04);

        vmovdqu64(zword [rcx], zmm29);
        mov(eax,0x55555555);
        kmovd(k1,eax);
        vmovdqu16(zword [rcx+0x2] | k1, zmm28);
        
        // vpslld(zmm28, zmm28, 0x10); // shift imag_res 16 bits left
        // // Set up writemask k1
        // mov(esi, 0xAAAAAAAA);
        // kmovd(k1, esi);
        // // Interleave real and imaginary
        // vmovdqu16(zmm29 | k1, zmm28);
        // // Write to memory
        // vmovdqa64(zword [rcx], zmm29);
        ret();
    }
};

Complex_float broad = {1.0,-1.0};
struct JitFloatMatVec : Xbyak::CodeGenerator {
    JitFloatMatVec(int m, int k)
        : Xbyak::CodeGenerator(4096, Xbyak::DontSetProtectRWE) // Use Read/Exec mode for security
    {  // Input parameters rdi={1, -1...}, rsi=mat, rdx=vec, rcx=res
        vpbroadcastq(zmm31, ptr [rdi]);
        vmovups(zmm30, zword [rsi]);
        vmovups(zmm29, zword [rsi+0x40]);
        mov(eax, 0x1F);
        vxorps(zmm26, zmm26, zmm26);
        vxorps(zmm25, zmm25, zmm25);
        vxorps(zmm24, zmm24, zmm24);
        vxorps(zmm23, zmm23, zmm23);
        vxorps(zmm22, zmm22, zmm22);
        vxorps(zmm21, zmm21, zmm21);
        vxorps(zmm20, zmm20, zmm20);
        vxorps(zmm19, zmm19, zmm19);
        L("L1");
            vmovups(zmm28, zword [rsi+0x80]);
            vmovups(zmm27, zword [rsi+0xC0]);
            vbroadcastss(zmm18, dword [rdx]);
            vfmadd231ps(zmm26, zmm30, zmm18);
            vfmadd231ps(zmm25, zmm29, zmm18);
            vbroadcastss(zmm18, dword [rdx+0x04]);
            vfmadd231ps(zmm22, zmm30, zmm18);
            vfmadd231ps(zmm21, zmm29, zmm18);
            vmovups(zmm30, zword [rsi+0x100]);
            vmovups(zmm29, zword [rsi+0x140]);
            vbroadcastss(zmm18, dword [rdx+0x08]);
            vfmadd231ps(zmm24, zmm28, zmm18);
            vfmadd231ps(zmm23, zmm27, zmm18);
            vbroadcastss(zmm18, dword [rdx+0x0C]);
            vfmadd231ps(zmm20, zmm28, zmm18);
            vfmadd231ps(zmm19, zmm27, zmm18);
            add(rsi, 0x100);
            add(rdx, 0x10);
            sub(rax, 0x01);
        jnle("L1");
        vmovups(zmm28, zword [rsi+0x80]);
        vmovups(zmm27, zword [rsi+0xC0]);
        vbroadcastss(zmm18, dword [rdx]);
        vfmadd231ps(zmm26, zmm30, zmm18);
        vfmadd231ps(zmm25, zmm29, zmm18);
        vbroadcastss(zmm18, dword [rdx+0x04]);
        vfmadd231ps(zmm22, zmm30, zmm18);
        vfmadd231ps(zmm21, zmm29, zmm18);
        vbroadcastss(zmm18, dword [rdx+0x08]);
        vfmadd231ps(zmm24, zmm28, zmm18);
        vfmadd231ps(zmm23, zmm27, zmm18);
        vbroadcastss(zmm18, dword [rdx+0x0C]);
        vfmadd231ps(zmm20, zmm28, zmm18);
        vfmadd231ps(zmm19, zmm27, zmm18);
        vaddps(zmm26, zmm26, zmm24);
        vaddps(zmm22, zmm22, zmm20);
        vaddps(zmm25, zmm25, zmm23);
        vaddps(zmm21, zmm21, zmm19);
        vpermilps(zmm22, zmm22, 0xB1);
        vfnmadd231ps(zmm26, zmm22, zmm31);
        vpermilps(zmm21, zmm21, 0xB1);
        vfnmadd231ps(zmm25, zmm21, zmm31);
        vmovups(zword [rcx], zmm26);
        vmovups(zword [rcx+0x40], zmm25);
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
    outputASM((void*)my_cgemm, m, k, "/asm/mkl");
    mkl_jit_destroy((void*)my_cgemm);
    return ret;
}

void matvecFloat_64x16(const MKL_Complex8* mat, const MKL_Complex8* vec, MKL_Complex8* res) {
    __m512 a_b, a_b1, a_b2, a_b3, a_b4, c_c, d_d, ac_bc, ad_bd, bd_ad, bd_ad1,bd_ad2,bd_ad3;
    __m512 ac_bc_accu, ad_bd_accu, ac_bc_accu1, ad_bd_accu1, ac_bc_accu2, ad_bd_accu2, ac_bc_accu3, ad_bd_accu3, ac_bc_accu4, ad_bd_accu4;
    MKL_Complex8 val;
    MKL_Complex8 broad = {1.0, -1.0};
    __m512 addSub = (__m512)_mm512_broadcastsd_pd(*(__m128d*)&broad);
    for(int r = 0; r < 64; r+=32) {
        a_b  = _mm512_loadu_ps((const void*)&mat[r+0]);
        a_b1 = _mm512_loadu_ps((const void*)&mat[r+8]);
        a_b2 = _mm512_loadu_ps((const void*)&mat[r+16]);
        a_b3 = _mm512_loadu_ps((const void*)&mat[r+24]);
        c_c = _mm512_set1_ps(vec[0].real);
        ac_bc_accu =  _mm512_mul_ps(a_b, c_c);
        ac_bc_accu1 = _mm512_mul_ps(a_b1, c_c);
        ac_bc_accu2 = _mm512_mul_ps(a_b2, c_c);
        ac_bc_accu3 = _mm512_mul_ps(a_b3, c_c);
        d_d = _mm512_set1_ps(vec[0].imag);
        ad_bd_accu =  _mm512_mul_ps(a_b, d_d);
        ad_bd_accu1 = _mm512_mul_ps(a_b1, d_d);
        ad_bd_accu2 = _mm512_mul_ps(a_b2, d_d);
        ad_bd_accu3 = _mm512_mul_ps(a_b3, d_d);
        for(int c = 1; c < 16; c++) {
            a_b1 = _mm512_loadu_ps((const void*)&mat[r+c*64+8]);
            a_b =  _mm512_loadu_ps((const void*)&mat[r+c*64]);
            a_b2 = _mm512_loadu_ps((const void*)&mat[r+c*64+16]);
            a_b3 = _mm512_loadu_ps((const void*)&mat[r+c*64+24]);
            c_c = _mm512_set1_ps(vec[c].real);
            ac_bc_accu =  _mm512_fmadd_ps(a_b, c_c, ac_bc_accu);
            ac_bc_accu1 = _mm512_fmadd_ps(a_b1, c_c, ac_bc_accu1);
            ac_bc_accu2 = _mm512_fmadd_ps(a_b2, c_c, ac_bc_accu2);
            ac_bc_accu3 = _mm512_fmadd_ps(a_b3, c_c, ac_bc_accu3);
            d_d = _mm512_set1_ps(vec[c].imag);
            ad_bd_accu =  _mm512_fmadd_ps(a_b, d_d, ad_bd_accu);
            ad_bd_accu1 = _mm512_fmadd_ps(a_b1, d_d, ad_bd_accu1);
            ad_bd_accu2 = _mm512_fmadd_ps(a_b2, d_d, ad_bd_accu2);
            ad_bd_accu3 = _mm512_fmadd_ps(a_b3, d_d, ad_bd_accu3);
        }
        bd_ad = _mm512_permute_ps(ad_bd_accu, 0xB1);
        ac_bc_accu = _mm512_fnmadd_ps(bd_ad, addSub, ac_bc_accu);
        bd_ad1 = _mm512_permute_ps(ad_bd_accu1, 0xB1);
        ac_bc_accu1 = _mm512_fnmadd_ps(bd_ad1, addSub, ac_bc_accu1);
        bd_ad2 = _mm512_permute_ps(ad_bd_accu2, 0xB1);
        ac_bc_accu2 = _mm512_fnmadd_ps(bd_ad2, addSub, ac_bc_accu2);
        bd_ad3 = _mm512_permute_ps(ad_bd_accu3, 0xB1);
        ac_bc_accu3 = _mm512_fnmadd_ps(bd_ad3, addSub, ac_bc_accu3);
        _mm512_storeu_ps(res+r, ac_bc_accu);
        _mm512_storeu_ps(res+r+8, ac_bc_accu1);
        _mm512_storeu_ps(res+r+16, ac_bc_accu2);
        _mm512_storeu_ps(res+r+24, ac_bc_accu3);
    }
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
        mat16[i] = {(int16_t)(i%mod-0+1), (int16_t)(i%mod-0)};
        mat[i] = {(float)mat16[i].real, (float)mat16[i].imag};
    }
    for(int i = 0; i < k; i++) {
        vec16[i] = {(int16_t)(i%mod-0), (int16_t)(i%mod-0+1)};
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

    // Generate code at runtime (Just-in-Time) and output asm
    double mklTime = runJITCGEMM((MKL_Complex8*)mat, (MKL_Complex8*)vec, (MKL_Complex8*)res1, m, k, numIter);

    double start = getTime();
    JitFloatMatVec jit(m, k);
    jit.setProtectModeRE(); // Use Read/Exec mode for security
    void (*matvec)(void* notUsed, const Complex_float*, const Complex_float*, Complex_float*) = jit.getCode<void (*)(void* notUsed, const Complex_float*, const Complex_float*, Complex_float*)>();
    for(int i = 0; i < numIter; i++) {
        matvec((void*)&broad, mat, vec, res);
    }
    double myFloatTime = timeSince(start);
    start = getTime();
    // for(int i = 0; i < numIter; i++)
    //     matvecFloat_64x16((MKL_Complex8*)mat, (MKL_Complex8*)vec, (MKL_Complex8*)res2);
    double oldFloatTime = timeSince(start);
    start = getTime();
    JitInt16MatVec jit16(m, k);
    jit16.setProtectModeRE(); // Use Read/Exec mode for security
    void (*matvec16)(void* notUsed, const Complex_int16*, const Complex_int16*, Complex_int16*, void* swapPairs) = jit16.getCode<void (*)(void* notUsed, const Complex_int16*, const Complex_int16*, Complex_int16*, void* swapPairs)>();
    for(int i = 0; i < numIter; i++)
        matvec16((void*)&broad16, mat16, vec16, res16, (void*)&swapPairs);
    double myTime = timeSince(start);

    // Save .asm of each function
    outputASM((void*)matvec, m, k, "./asm/myfloat");
    outputASM((void*)matvec16, m, k, "./asm/myint16");

    // Output result
    for(int i = 0; i < m; i++) std::cout << res[i];
    std::cout << std::endl;
    for(int i = 0; i < m; i++) std::cout << res1[i];
    std::cout << std::endl;
    // for(int i = 0; i < m; i++) std::cout << res2[i];
    // std::cout << std::endl;
    for(int i = 0; i < m; i++) std::cout << res16[i];
    std::cout << std::endl;


    printf("\n        ---------- \n\n");
    printf("     %ld iterations, (%ldx%ld) * (%ldx%d)\n", numIter, m, k, k, 1);
    printf("MKL JIT cgemm: %.10f µs per iteration\n", mklTime/(double)numIter);
    printf(" my JIT float: %.10f µs per iteration\n", myFloatTime/(double)numIter);
    printf("    old float: %.10f µs per iteration\n", oldFloatTime/(double)numIter);
    printf(" my JIT int16: %.10f µs per iteration\n", myTime/(double)numIter);
    // // Free allocated memory
    // free(mat); free(vec); free(res); free(res1); free(res2);
    // free(mat16); free(vec16); free(res16);
}