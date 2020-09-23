# Fixed-point, integer, complex matrix vector multiplication

byl_jit_int16_matvec.hpp provides an int16_t, fixed-point, complex matrix vector multiplication kernels using the Xbyak just-in-time code generator. It runs ~1.5 - 6x faster than Intel MKL's JIT cgemm. It uses fixed-point arithmetic as opposed to floating point so as to take up half the memory compared to single-precision float and thus allows for twice as many computations per SIMD instruction. With 9 fractional bits, the range and precision is sufficient for the input values in the 5G baseband research project that I developed this for at the Yale Efficient Computing Lab. I use AVX-512 instructions in my kernel generator including vpdpwssds from the Vector Neural Network Instruction extension for integer fused-multiply add operations. Currently, my kernel generator supports matrix dimensions M rows by K columns where M <= 208, M is a multiple of 16, and any K.

1. Install Xbyak acorrding to the [instructions on Github](https://github.com/herumi/xbyak#install)
2. Copy byl_jit_int16_matvec.hpp to your project directory and include it in any files necessary
3. Create and use a complex matric vector multiplication kernel for MxK * Kx1 problem size referencing the following sample code:

```c++
// "jit16" and "matvec16" are placeholder names and can be set by the user
JitInt16MatVec jit16(m, k); // Initialize the Xbyak CodeGenerator for problem size MxK * Kx1
jit16.readyRE(); // Set the jitted memory region as read/execute only for security
void (*matvec16)(const Complex_int16*, const Complex_int16*, Complex_int16*) = jit16.getCode<void (*)(const Complex_int16*, const Complex_int16*, Complex_int16*)>(); // Store the function pointer in "matvec16"
matvec16(mat16, vec16, res16); // Perform complex matrix multiplcation on mat16 * vect16 = res16
```
Note that `Complex_int16` is defined in byl_jit_int16_matvec.hpp and complex numbers are stored in interleaved format. Also note that the values of res16 must be initially zero. Only one kernel can be generated per function at this time. An example program generating and running two kernels for different M and K on random real numbers and printing the output can be found in main.cpp. To change the number of fractional bits in the fixed point representation to adjust the precision/range of values, simply change the line `#define FIXED_POINT_FRACTIONAL_BITS` to the value desired.


