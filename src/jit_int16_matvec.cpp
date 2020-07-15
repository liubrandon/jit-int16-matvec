#include <xbyak/xbyak.h>

struct Code : Xbyak::CodeGenerator {
    Code(int x) {
        mov(eax, x);
        ret();
    }
};

int main() {
    Code c(5);
    int (*f)() = c.getCode<int (*)()>();
    printf("ret=%d\n", f()); //
    return 0;
}