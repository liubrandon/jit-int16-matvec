#include <xbyak/xbyak.h>
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

struct JitInt16MatVec : Xbyak::CodeGenerator {
    JitInt16MatVec(int m, int k)
        : Xbyak::CodeGenerator(4096, Xbyak::DontSetProtectRWE) // use Read/Exec mode for security
    {  // Input parameters rdi=mat, rsi=vec, rdx=res (https://aaronbloomfield.github.io/pdr/book/x86-64bit-ccc-chapter.pdf)
        mov(eax, dword [rdi]); // This returns the first real element of mat
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
    for(int i = 0; i < k; i++)   vec[i] = {(int16_t)(i+1), (int16_t)(0)};

    // Generate code at runtime (Just-in-Time)
    JitInt16MatVec jit(m, k);
    jit.setProtectModeRE(); // use Read/Exec mode for security
    Complex_int16 (*matvec)(const Complex_int16*, const Complex_int16*, Complex_int16*) = jit.getCode<Complex_int16 (*)(const Complex_int16*, const Complex_int16*, Complex_int16*)>();

    // Output result
    std::cout << matvec(mat, vec, res) << std::endl;

    // Free allocated memory
    free(mat); free(vec); free(res);
}