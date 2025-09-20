### To launch this kernel on my GPU, which is NVIDIA RTX A5000 Laptop GPU with compute capability: 8.6.
The command to compile the cuda code is:
```bash
$ nvcc -arch=compute_86 -code=sm_86,compute_86 -o out SGEMM_naive.cu
```
To run the full Nsight Compute (ncu) analysis, the following is the command:
```bash
ncu --kernel-name sgemm_naive --set full ./out
```
