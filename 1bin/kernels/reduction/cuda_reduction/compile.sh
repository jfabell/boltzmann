module load gcc/15.1.0

nvcc -arch=sm_90a kernels-new.cu -O3 \
 -Xcompiler -fPIC \
 `python3 -m pybind11 --includes` \
 -shared -o cuda_red`python3.9-config --extension-suffix` \
 -L.
