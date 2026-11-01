// Scoped runtime workaround for the tested HPC-X 2.50 CUDA initialization order.
// Require one GPU visible per process; do not infer device placement here.
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
static void initialize_cuda() {
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess || count != 1) {
    std::fprintf(stderr, "CUDA pre-MPI initialization requires exactly one visible GPU (count=%d, error=%d)\n", count, int(e));
    std::exit(77);
  }
  e = cudaSetDevice(0);
  if (e == cudaSuccess) e = cudaFree(nullptr);
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA pre-MPI initialization failed: %s\n", cudaGetErrorString(e));
    std::exit(78);
  }
}
extern "C" int MPI_Init(int *argc, char ***argv) { initialize_cuda(); return PMPI_Init(argc, argv); }
extern "C" int MPI_Init_thread(int *argc, char ***argv, int requested, int *provided) { initialize_cuda(); return PMPI_Init_thread(argc, argv, requested, provided); }
