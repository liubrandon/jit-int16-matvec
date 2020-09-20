#include "byl_jit_int16_matvec.hpp"

void testMultipleRuns(long m, long k) {
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
    std::uniform_real_distribution<float> mat_dis(-1.0, 1.0);
    int mod = 10; //rand()%mod-range
    int range = mod/2;
    for(int i = 0; i < m*k; i++) {
        mat[i] = {mat_dis(gen), mat_dis(gen)};
        mat16[i] = {floatToFixed(mat[i].real), floatToFixed(mat[i].imag)};
    }
    std::uniform_real_distribution<float> vec_dis(-5.0, 5.0);
    for(int i = 0; i < k; i++) {
        vec[i] = {vec_dis(gen), vec_dis(gen)};
        vec16[i] = {floatToFixed(vec[i].real), floatToFixed(vec[i].imag)};
    }
    JitInt16MatVec jit16(m, k);
    jit16.readyRE();
    void (*matvec16)(const Complex_int16*, const Complex_int16*, Complex_int16*) = jit16.getCode<void (*)(const Complex_int16*, const Complex_int16*, Complex_int16*)>();
    matvec16(mat16, vec16, res16);
    MKL_Complex8 alpha = {1, 0};
    MKL_Complex8 beta = {0, 0};
    MKL_INT lda = m;
    MKL_INT ldb = k;
    MKL_INT ldc = m;
    // Create a handle and generate GEMM kernel
    void* jitter;
    mkl_jit_status_t status = mkl_jit_create_cgemm(&jitter, MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, m, 1, k, &alpha, lda, ldb, &beta, ldc);
    if (MKL_JIT_ERROR == status) {
        fprintf(stderr, "Error: insufficient memory to JIT and store the DGEMM kernel\n");
        exit(1);
    }
    // Get kernel associated with handle
    cgemm_jit_kernel_t my_cgemm = mkl_jit_get_cgemm_ptr(jitter);
    my_cgemm(jitter, mat, vec, res); // Repeatedly execute the GEMM kernel
    // Destroy the created handle/GEMM kernel
    mkl_jit_destroy((void*)my_cgemm);
    // Output result
    std::cout << "1: " << std::endl;
    for(int i = 0; i < m; i++) std::cout << "(" << std::fixed << std::setprecision(2) << res[i].real << "," << std::fixed << std::setprecision(2) << res[i].imag << ")";
    std::cout << std::endl;
    std::cout << "2: " << std::endl;
    for(int i = 0; i < m; i++) {
        std::cout << "(" << std::fixed << std::setprecision(2) << fixedToFloat(res16[i].real) << "," << std::fixed << std::setprecision(2) << fixedToFloat(res16[i].imag) << ")";
    }
    std::cout << std::endl;
    // Free allocated memory
    mkl_free(mat); mkl_free(vec); mkl_free(res);
    free(mat16); free(vec16); free(res16);
    
}

int main() {
    testMultipleRuns(64,16);
    testMultipleRuns(16,64);

    // MKL_Complex8 *mat_other, *vec_other, *res_other;
    // Complex_int16 *mat16_other, *vec16_other, *res16_other;
    // long m_other = 16;
    // long k_other = 64;
    // // Allocate memory for matrix, vector, and resulting vector aligned on 64 byte boundary
    // mat_other = (MKL_Complex8*)mkl_calloc(m_other*k_other, sizeof(MKL_Complex8), 64);
    // vec_other = (MKL_Complex8*)mkl_calloc(k_other, sizeof(MKL_Complex8), 64);
    // res_other = (MKL_Complex8*)mkl_calloc(m_other, sizeof(MKL_Complex8), 64);
    // // Int16 version
    // mat16_other = (Complex_int16*)aligned_alloc(128, m_other*k*sizeof(Complex_int16));
    // vec16_other = (Complex_int16*)aligned_alloc(128, k_other*sizeof(Complex_int16));
    // res16_other = (Complex_int16*)aligned_alloc(128, m_other*sizeof(Complex_int16));
    // memset(res16_other, 0, m_other*sizeof(Complex_int16));
    // // Randomly generate matrix/vector with values from 0 to 50
    // for(int i = 0; i < m_other*k_other; i++) {
    //     mat_other[i] = {mat_dis(gen), mat_dis(gen)};
    //     mat16_other[i] = {floatToFixed(mat_other[i].real), floatToFixed(mat_other[i].imag)};
    // }
    // for(int i = 0; i < k_other; i++) {
    //     vec_other[i] = {vec_dis(gen), vec_dis(gen)};
    //     vec16_other[i] = {floatToFixed(vec_other[i].real), floatToFixed(vec_other[i].imag)};
    // }
    // JitInt16MatVec jit16_other(m_other, k_other);
    // // jit16_other.ready();
    // // jit16_other.setProtectModeRE(); // Use Read/Exec mode for security
    // void (*matvec16_other)(const Complex_int16*, const Complex_int16*, Complex_int16*) = jit16_other.getCode<void (*)(const Complex_int16*, const Complex_int16*, Complex_int16*)>();
    // matvec16_other(mat16_other, vec16_other, res16_other);
    // jit16_other.resetSize();
    // jit16_other.reset();
    // // Create a handle and generate GEMM kernel
    // void* jitter_other;
    // mkl_jit_status_t status_other = mkl_jit_create_cgemm(&jitter_other, MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, m, 1, k, &alpha, lda, ldb, &beta, ldc);
    // if (MKL_JIT_ERROR == status_other) {
    //     fprintf(stderr, "Error: insufficient memory to JIT and store the DGEMM kernel\n");
    //     exit(1);
    // }
    // // Get kernel associated with handle
    // cgemm_jit_kernel_t my_cgemm_other = mkl_jit_get_cgemm_ptr(jitter_other);
    // my_cgemm_other(jitter_other, mat_other, vec_other, res_other); // Repeatedly execute the GEMM kernel
    // // Destroy the created handle/GEMM kernel
    // mkl_jit_destroy((void*)my_cgemm_other);
    // // Output result
    // std::cout << "3: " << std::endl;
    // for(int i = 0; i < m_other; i++) std::cout << "(" << std::fixed << std::setprecision(2) << res[i].real << "," << std::fixed << std::setprecision(2) << res[i].imag << ")";
    // std::cout << std::endl;
    // std::cout << "4: " << std::endl;
    // for(int i = 0; i < m_other; i++) {
    //     std::cout << "(" << std::fixed << std::setprecision(2) << fixedToFloat(res16[i].real) << "," << std::fixed << std::setprecision(2) << fixedToFloat(res16[i].imag) << ")";
    // }
}