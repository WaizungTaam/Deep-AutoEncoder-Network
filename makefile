CC=g++
FLAG=-std=c++11

all: deep_ae_test_binary.o deep_ae_test_continuous.o

deep_ae_test_binary.o: ./src/math/* ./src/deep_ae.h ./src/deep_ae.cc ./test/deep_ae_test_binary.cc
	$(CC) $(FLAG) ./src/math/* ./src/deep_ae.h ./src/deep_ae.cc ./test/deep_ae_test_binary.cc -o deep_ae_test_binary.o

deep_ae_test_continuous.o: ./src/math/* ./src/deep_ae.h ./src/deep_ae.cc ./test/deep_ae_test_continuous.cc
	$(CC) $(FLAG) ./src/math/* ./src/deep_ae.h ./src/deep_ae.cc ./test/deep_ae_test_continuous.cc -o deep_ae_test_continuous.o

clean:
	rm *.o