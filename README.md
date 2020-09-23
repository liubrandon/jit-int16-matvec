# Fixed-point, integer, complex matrix vector multiplication

byl_jit_int16_matvec.hpp provides an int16_t, fixed-point, complex matrix vector multiplication kernels using the Xbyak just-in-time code generator. It runs ~1.5 - 6x faster than Intel MKL's JIT cgemm.


### Implementation details and limitations
I use AVX-512 instructions in my kernel generator including vpdpwssds from the Vector Neural Network Instruction extension for integer fused-multiply add operations. Currently, my kernel generator supports matrix dimensions M rows by K columns where M <= 208, M is a multiple of 16, and any K.

The just-in-time compiled kernels use fixed-point arithmetic on int16_t as opposed to floating-point arithmetic on floats (32 bits) so as to take up half the memory space, allow for twice as many computations per SIMD instruction, and reduce memory loads and stores by half. With 9 fractional bits, the range and precision is sufficient for the input values in the 5G software baseband processing research project that I developed this for at the Yale Efficient Computing Lab.

### Instructions
1. Install Xbyak according to the [instructions on Github](https://github.com/herumi/xbyak#install)
2. Copy byl_jit_int16_matvec.hpp to your project directory and include it in any files necessary
3. Create and use a complex matrix vector multiplication kernel for a MxK * Kx1 problem size referencing the following sample code:

```c++
// "jit16" and "matvec16" are placeholder names and can be set by the user
JitInt16MatVec jit16(m, k); // Initialize the Xbyak CodeGenerator for problem size MxK * Kx1
jit16.readyRE(); // Set the jitted memory region as read/execute only for security
void (*matvec16)(const Complex_int16*, const Complex_int16*, Complex_int16*) = jit16.getCode<void (*)(const Complex_int16*, const Complex_int16*, Complex_int16*)>(); // Store the function pointer in "matvec16"
matvec16(mat16, vec16, res16); // Perform complex matrix multiplcation on mat16 * vect16 = res16
```
### Notes:
* `Complex_int16` is defined in byl_jit_int16_matvec.hpp and complex numbers are stored in interleaved format.
* The memory region occupied by the resultant vector (res16 in the example) must be initially zero.
* Only one kernel can be generated per function at this time. 
* An example program generating and running two kernels for different M and K on random real numbers and printing the output can be found in main.cpp. 
* To change the number of fractional bits in the fixed point representation to adjust the precision/range of values, simply change the `#define FIXED_POINT_FRACTIONAL_BITS` to the value desired.



