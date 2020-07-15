#include <xbyak/xbyak.h>
#include <Zydis/Zydis.h>
#include <fstream>
#include <string.h>
#include <iostream>
#define DIE(...) fprintf(stderr, __VA_ARGS__); exit(1);

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

int main(int argc, char** argv) {
    // Parse arguments and declare/initialize variables
    if(argc != 2) {
        DIE("Usage: %s MxK where M is # rows and K is # cols\n", argv[0]);
    }
    srand(time(0));
    long numIter = 100000;
    int m, k;
    Complex_int16 *mat, *vec, *res;
    char* nPtr;
    m = strtoul(argv[1], &nPtr, 0);
    k = strtoul(nPtr+1, NULL, 0);

    // Allocate memory for matrix, vector, and resulting vector aligned on 64 byte boundary
    mat = (Complex_int16*)aligned_alloc(64, m*k*sizeof(Complex_int16));
    vec = (Complex_int16*)aligned_alloc(64, k*sizeof(Complex_int16));
    res = (Complex_int16*)aligned_alloc(64, m*sizeof(Complex_int16));
    memset(res, 0, m*sizeof(Complex_int16));

    // Randomly generate matrix/vector with values from -range to range
    int mod = 10; //rand()%mod-range
    int range = mod/2;
    for(int i = 0; i < m*k; i++) mat[i] = {(int16_t)(-100), (int16_t)(i+1)};
    for(int i = 0; i < k; i++)   vec[i] = {(int16_t)(i+1), (int16_t)(i)};

    // Generate code at runtime (Just-in-Time) and output asm
    JitInt16MatVec jit(m, k);
    jit.setProtectModeRE(); // Use Read/Exec mode for security
    void (*matvec)(const Complex_int16*, const Complex_int16*, Complex_int16*) = jit.getCode<void (*)(const Complex_int16*, const Complex_int16*, Complex_int16*)>();
    outputASM((void*)matvec, m, k, "jit");

    matvec(mat, vec, res);

    // Output result
    for(int i = 0; i < m; i++) std::cout << res[i];
    std::cout << std::endl;

    // Free allocated memory
    free(mat); free(vec); free(res);
}