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
    {  // Input parameters rdi={1, -1...}, rsi=mat, rdx=vec, rcx=res
        vpbroadcastq(zmm31, ptr [rdi]);
        vmovups(zmm30, zword [rsi]);
        vmovups(zmm29, zword [rsi+0x40]);
        vmovups(zmm28, zword [rsi+0x80]);
        vmovups(zmm27, zword [rsi+0xC0]);
        vmovups(zmm26, zword [rsi+0x200]);
        vmovups(zmm25, zword [rsi+0x240]);
        vmovups(zmm24, zword [rsi+0x280]);
        vmovups(zmm23, zword [rsi+0x2C0]);
        vbroadcastss(zmm14, dword [rdx]);
        vmulps(zmm22, zmm30, zmm14);
        vmulps(zmm21, zmm29, zmm14);
        vmulps(zmm20, zmm28, zmm14);
        vmulps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x04]);
        vmulps(zmm18, zmm30, zmm14);
        vmulps(zmm17, zmm29, zmm14);
        vmulps(zmm16, zmm28, zmm14);
        vmulps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0x400]);
        vmovups(zmm29, zword [rsi+0x440]);
        vmovups(zmm28, zword [rsi+0x480]);
        vmovups(zmm27, zword [rsi+0x4C0]);
        vbroadcastss(zmm14, dword [rdx+0x08]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x0C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0x600]);
        vmovups(zmm25, zword [rsi+0x640]);
        vmovups(zmm24, zword [rsi+0x680]);
        vmovups(zmm23, zword [rsi+0x6C0]);
        vbroadcastss(zmm14, dword [rdx+0x10]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x14]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0x800]);
        vmovups(zmm29, zword [rsi+0x840]);
        vmovups(zmm28, zword [rsi+0x880]);
        vmovups(zmm27, zword [rsi+0x8C0]);
        vbroadcastss(zmm14, dword [rdx+0x18]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x1C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0xA00]);
        vmovups(zmm25, zword [rsi+0xA40]);
        vmovups(zmm24, zword [rsi+0xA80]);
        vmovups(zmm23, zword [rsi+0xAC0]);
        vbroadcastss(zmm14, dword [rdx+0x20]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x24]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0xC00]);
        vmovups(zmm29, zword [rsi+0xC40]);
        vmovups(zmm28, zword [rsi+0xC80]);
        vmovups(zmm27, zword [rsi+0xCC0]);
        vbroadcastss(zmm14, dword [rdx+0x28]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x2C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0xE00]);
        vmovups(zmm25, zword [rsi+0xE40]);
        vmovups(zmm24, zword [rsi+0xE80]);
        vmovups(zmm23, zword [rsi+0xEC0]);
        vbroadcastss(zmm14, dword [rdx+0x30]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x34]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0x1000]);
        vmovups(zmm29, zword [rsi+0x1040]);
        vmovups(zmm28, zword [rsi+0x1080]);
        vmovups(zmm27, zword [rsi+0x10C0]);
        vbroadcastss(zmm14, dword [rdx+0x38]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x3C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0x1200]);
        vmovups(zmm25, zword [rsi+0x1240]);
        vmovups(zmm24, zword [rsi+0x1280]);
        vmovups(zmm23, zword [rsi+0x12C0]);
        vbroadcastss(zmm14, dword [rdx+0x40]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x44]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0x1400]);
        vmovups(zmm29, zword [rsi+0x1440]);
        vmovups(zmm28, zword [rsi+0x1480]);
        vmovups(zmm27, zword [rsi+0x14C0]);
        vbroadcastss(zmm14, dword [rdx+0x48]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x4C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0x1600]);
        vmovups(zmm25, zword [rsi+0x1640]);
        vmovups(zmm24, zword [rsi+0x1680]);
        vmovups(zmm23, zword [rsi+0x16C0]);
        vbroadcastss(zmm14, dword [rdx+0x50]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x54]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0x1800]);
        vmovups(zmm29, zword [rsi+0x1840]);
        vmovups(zmm28, zword [rsi+0x1880]);
        vmovups(zmm27, zword [rsi+0x18C0]);
        vbroadcastss(zmm14, dword [rdx+0x58]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x5C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0x1A00]);
        vmovups(zmm25, zword [rsi+0x1A40]);
        vmovups(zmm24, zword [rsi+0x1A80]);
        vmovups(zmm23, zword [rsi+0x1AC0]);
        vbroadcastss(zmm14, dword [rdx+0x60]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x64]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0x1C00]);
        vmovups(zmm29, zword [rsi+0x1C40]);
        vmovups(zmm28, zword [rsi+0x1C80]);
        vmovups(zmm27, zword [rsi+0x1CC0]);
        vbroadcastss(zmm14, dword [rdx+0x68]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x6C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0x1E00]);
        vmovups(zmm25, zword [rsi+0x1E40]);
        vmovups(zmm24, zword [rsi+0x1E80]);
        vmovups(zmm23, zword [rsi+0x1EC0]);
        vbroadcastss(zmm14, dword [rdx+0x70]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x74]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x78]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x7C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vpermilps(zmm18, zmm18, 0xB1);
        vfnmadd231ps(zmm22, zmm18, zmm31);
        vpermilps(zmm17, zmm17, 0xB1);
        vfnmadd231ps(zmm21, zmm17, zmm31);
        vpermilps(zmm16, zmm16, 0xB1);
        vfnmadd231ps(zmm20, zmm16, zmm31);
        vpermilps(zmm15, zmm15, 0xB1);
        vfnmadd231ps(zmm19, zmm15, zmm31);
        vmovups(zword [rcx], zmm22);
        vmovups(zword [rcx+0x40], zmm21);
        vmovups(zword [rcx+0x80], zmm20);
        vmovups(zword [rcx+0xC0], zmm19);
        vmovups(zmm30, zword [rsi+0x100]);
        vmovups(zmm29, zword [rsi+0x140]);
        vmovups(zmm28, zword [rsi+0x180]);
        vmovups(zmm27, zword [rsi+0x1C0]);
        vmovups(zmm26, zword [rsi+0x300]);
        vmovups(zmm25, zword [rsi+0x340]);
        vmovups(zmm24, zword [rsi+0x380]);
        vmovups(zmm23, zword [rsi+0x3C0]);
        vbroadcastss(zmm14, dword [rdx]);
        vmulps(zmm22, zmm30, zmm14);
        vmulps(zmm21, zmm29, zmm14);
        vmulps(zmm20, zmm28, zmm14);
        vmulps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x04]);
        vmulps(zmm18, zmm30, zmm14);
        vmulps(zmm17, zmm29, zmm14);
        vmulps(zmm16, zmm28, zmm14);
        vmulps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0x500]);
        vmovups(zmm29, zword [rsi+0x540]);
        vmovups(zmm28, zword [rsi+0x580]);
        vmovups(zmm27, zword [rsi+0x5C0]);
        vbroadcastss(zmm14, dword [rdx+0x08]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x0C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0x700]);
        vmovups(zmm25, zword [rsi+0x740]);
        vmovups(zmm24, zword [rsi+0x780]);
        vmovups(zmm23, zword [rsi+0x7C0]);
        vbroadcastss(zmm14, dword [rdx+0x10]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x14]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0x900]);
        vmovups(zmm29, zword [rsi+0x940]);
        vmovups(zmm28, zword [rsi+0x980]);
        vmovups(zmm27, zword [rsi+0x9C0]);
        vbroadcastss(zmm14, dword [rdx+0x18]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x1C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0xB00]);
        vmovups(zmm25, zword [rsi+0xB40]);
        vmovups(zmm24, zword [rsi+0xB80]);
        vmovups(zmm23, zword [rsi+0xBC0]);
        vbroadcastss(zmm14, dword [rdx+0x20]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x24]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0xD00]);
        vmovups(zmm29, zword [rsi+0xD40]);
        vmovups(zmm28, zword [rsi+0xD80]);
        vmovups(zmm27, zword [rsi+0xDC0]);
        vbroadcastss(zmm14, dword [rdx+0x28]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x2C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0xF00]);
        vmovups(zmm25, zword [rsi+0xF40]);
        vmovups(zmm24, zword [rsi+0xF80]);
        vmovups(zmm23, zword [rsi+0xFC0]);
        vbroadcastss(zmm14, dword [rdx+0x30]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x34]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0x1100]);
        vmovups(zmm29, zword [rsi+0x1140]);
        vmovups(zmm28, zword [rsi+0x1180]);
        vmovups(zmm27, zword [rsi+0x11C0]);
        vbroadcastss(zmm14, dword [rdx+0x38]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x3C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0x1300]);
        vmovups(zmm25, zword [rsi+0x1340]);
        vmovups(zmm24, zword [rsi+0x1380]);
        vmovups(zmm23, zword [rsi+0x13C0]);
        vbroadcastss(zmm14, dword [rdx+0x40]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x44]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0x1500]);
        vmovups(zmm29, zword [rsi+0x1540]);
        vmovups(zmm28, zword [rsi+0x1580]);
        vmovups(zmm27, zword [rsi+0x15C0]);
        vbroadcastss(zmm14, dword [rdx+0x48]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x4C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0x1700]);
        vmovups(zmm25, zword [rsi+0x1740]);
        vmovups(zmm24, zword [rsi+0x1780]);
        vmovups(zmm23, zword [rsi+0x17C0]);
        vbroadcastss(zmm14, dword [rdx+0x50]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x54]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0x1900]);
        vmovups(zmm29, zword [rsi+0x1940]);
        vmovups(zmm28, zword [rsi+0x1980]);
        vmovups(zmm27, zword [rsi+0x19C0]);
        vbroadcastss(zmm14, dword [rdx+0x58]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x5C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0x1B00]);
        vmovups(zmm25, zword [rsi+0x1B40]);
        vmovups(zmm24, zword [rsi+0x1B80]);
        vmovups(zmm23, zword [rsi+0x1BC0]);
        vbroadcastss(zmm14, dword [rdx+0x60]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x64]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vmovups(zmm30, zword [rsi+0x1D00]);
        vmovups(zmm29, zword [rsi+0x1D40]);
        vmovups(zmm28, zword [rsi+0x1D80]);
        vmovups(zmm27, zword [rsi+0x1DC0]);
        vbroadcastss(zmm14, dword [rdx+0x68]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x6C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vmovups(zmm26, zword [rsi+0x1F00]);
        vmovups(zmm25, zword [rsi+0x1F40]);
        vmovups(zmm24, zword [rsi+0x1F80]);
        vmovups(zmm23, zword [rsi+0x1FC0]);
        vbroadcastss(zmm14, dword [rdx+0x70]);
        vfmadd231ps(zmm22, zmm30, zmm14);
        vfmadd231ps(zmm21, zmm29, zmm14);
        vfmadd231ps(zmm20, zmm28, zmm14);
        vfmadd231ps(zmm19, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x74]);
        vfmadd231ps(zmm18, zmm30, zmm14);
        vfmadd231ps(zmm17, zmm29, zmm14);
        vfmadd231ps(zmm16, zmm28, zmm14);
        vfmadd231ps(zmm15, zmm27, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x78]);
        vfmadd231ps(zmm22, zmm26, zmm14);
        vfmadd231ps(zmm21, zmm25, zmm14);
        vfmadd231ps(zmm20, zmm24, zmm14);
        vfmadd231ps(zmm19, zmm23, zmm14);
        vbroadcastss(zmm14, dword [rdx+0x7C]);
        vfmadd231ps(zmm18, zmm26, zmm14);
        vfmadd231ps(zmm17, zmm25, zmm14);
        vfmadd231ps(zmm16, zmm24, zmm14);
        vfmadd231ps(zmm15, zmm23, zmm14);
        vpermilps(zmm18, zmm18, 0xB1);
        vfnmadd231ps(zmm22, zmm18, zmm31);
        vpermilps(zmm17, zmm17, 0xB1);
        vfnmadd231ps(zmm21, zmm17, zmm31);
        vpermilps(zmm16, zmm16, 0xB1);
        vfnmadd231ps(zmm20, zmm16, zmm31);
        vpermilps(zmm15, zmm15, 0xB1);
        vfnmadd231ps(zmm19, zmm15, zmm31);
        vmovups(zword [rcx+0x100], zmm22);
        vmovups(zword [rcx+0x140], zmm21);
        vmovups(zword [rcx+0x180], zmm20);
        vmovups(zword [rcx+0x1C0], zmm19);
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

// static const float temp3[16] = {1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1};
// const __m512 addSub = _mm512_loadu_ps((const void*)temp3);
void matvecFloat_16x16(const MKL_Complex8* mat, const MKL_Complex8* vec, MKL_Complex8* res) {
    __m512 a_b, a_b1, a_b2, a_b3, c_c, d_d, ac_bc, ad_bd, bd_ad, bd_ad1;
    __m512 ac_bc_accu, ad_bd_accu, ac_bc_accu1, ad_bd_accu1, ac_bc_accu2, ad_bd_accu2, ac_bc_accu3, ad_bd_accu3, ac_bc_accu4, ad_bd_accu4;
    MKL_Complex8 val;
    MKL_Complex8 broad = {1.0, -1.0};
    __m512 addSub = (__m512)_mm512_broadcastsd_pd(*(__m128d*)&broad);
    val = vec[0];
    a_b = _mm512_loadu_ps((const void*)&mat[0]);
    a_b1 = _mm512_loadu_ps((const void*)&mat[8]);
    c_c = _mm512_set1_ps(val.real);
    ac_bc_accu = _mm512_mul_ps(a_b, c_c);
    ac_bc_accu1 = _mm512_mul_ps(a_b1, c_c);
    d_d = _mm512_set1_ps(val.imag);
    ad_bd_accu = _mm512_mul_ps(a_b, d_d);
    ad_bd_accu1 = _mm512_mul_ps(a_b1, d_d);

    for(int c = 1; c < 16; c++) {
        val = vec[c];
        a_b = _mm512_loadu_ps((const void*)&mat[c*16]);
        a_b1 = _mm512_loadu_ps((const void*)&mat[c*16+8]);
        c_c = _mm512_set1_ps(val.real);
        ac_bc_accu = _mm512_fmadd_ps(a_b, c_c, ac_bc_accu);
        ac_bc_accu1 = _mm512_fmadd_ps(a_b1, c_c, ac_bc_accu1);
        d_d = _mm512_set1_ps(val.imag);
        ad_bd_accu = _mm512_fmadd_ps(a_b, d_d, ad_bd_accu);
        ad_bd_accu1 = _mm512_fmadd_ps(a_b1, d_d, ad_bd_accu1);
    }
    // Swap even and odd in ad_bd
    bd_ad = _mm512_permute_ps(ad_bd_accu, 0xB1);
    ac_bc_accu = _mm512_fnmadd_ps(bd_ad, addSub, ac_bc_accu);

    bd_ad1 = _mm512_permute_ps(ad_bd_accu1, 0xB1);

    // Fused negate multiply add -> simplified to multiply, add
    ac_bc_accu1 = _mm512_fnmadd_ps(bd_ad1, addSub, ac_bc_accu1);

    // Store results in the array
    _mm512_storeu_ps(res, ac_bc_accu);
    _mm512_storeu_ps(res+8, ac_bc_accu1);
}

int main(int argc, char** argv) {
    // Parse arguments and declare/initialize variables
    if(argc != 2) {
        DIE("Usage: %s MxK where M is # rows and K is # cols\n", argv[0]);
    }
    srand(time(0));
    long numIter = 10000000;
    int m, k;
    Complex_float *mat, *vec, *res, *res1, *res2;
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
    res2 = (Complex_float*)aligned_alloc(64, m*sizeof(Complex_float));
    memset(res2, 0, m*sizeof(Complex_float));

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
    for(int i = 0; i < numIter; i++) {
        //matvec((void*)&broad, mat, vec, res);
    }
    double myTime = timeSince(start);
    double mklTime = runJITCGEMM((MKL_Complex8*)mat, (MKL_Complex8*)vec, (MKL_Complex8*)res1, m, k, numIter);
    start = getTime();
    for(int i = 0; i < numIter; i++)
        matvecFloat_16x16((MKL_Complex8*)mat, (MKL_Complex8*)vec, (MKL_Complex8*)res2);
    double oldTime = timeSince(start);

    // Output result
    for(int i = 0; i < m; i++) std::cout << res[i];
    std::cout << std::endl;
    for(int i = 0; i < m; i++) std::cout << res1[i];
    std::cout << std::endl;
    for(int i = 0; i < m; i++) std::cout << res2[i];
    std::cout << std::endl;
    outputASM((void*)matvec, m, k, "myfloat");

    printf("\n        ---------- \n\n");
    printf("     %ld iterations, (%dx%d) * (%dx%d)\n", numIter, m, k, k, 1);
    printf("MKL JIT cgemm: %.10f µs per iteration\n", mklTime/(double)numIter);
    printf(" my JIT float: %.10f µs per iteration\n", myTime/(double)numIter);
    printf("    old float: %.10f µs per iteration\n", oldTime/(double)numIter);
    // Free allocated memory
    free(mat); free(vec); free(res);
}