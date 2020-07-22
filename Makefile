.PHONY: matvec
matvec:
	g++ -g -m64 -I${MKLROOT}/include -o matvec ./src/jit_int16_matvec.cpp -I/usr/local/include/Zydis /usr/local/lib/libZydis.a -Wl,--no-as-needed -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group,--no-as-needed -lpthread -lm -ldl -march=native -O3