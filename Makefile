build/hello: src/hello.cu
	nvcc src/hello.cu -o build/hello -arch=compute_50

clean:
	rm -f build/hello
