build/hello: src/hello.cu
	nvcc src/hello.cu -o build/hello

clean:
	rm -f build/hello
