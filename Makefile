.PHONY: matvec main pow
pow:
	g++ -g -shared-libgcc -m64 -I${MKLROOT}/include -o pow ./src/byl_jit_int16_matvec.cpp -I/usr/local/include/Zydis /usr/local/lib/libZydis.a -Wl,--no-as-needed -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group,--no-as-needed -lpthread -lm -ldl -march=native
matvec:
	g++ -g -shared-libgcc -m64 -I${MKLROOT}/include -o matvec ./src/byl_jit_int16_matvec.cpp -I/usr/local/include/Zydis /usr/local/lib/libZydis.a -Wl,--no-as-needed -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group,--no-as-needed -lpthread -lm -ldl -march=native

main:
	g++ -shared-libgcc -m64 -I${MKLROOT}/include -o main ./src/main.cpp -Wl,--no-as-needed -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group,--no-as-needed -lpthread -lm -ldl -march=native