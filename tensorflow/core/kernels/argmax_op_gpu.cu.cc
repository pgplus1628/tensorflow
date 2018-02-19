/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/argmax_op.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPEC(T)                              \
  template struct functor::ArgMax<GPUDevice, T, int64>; \
  template struct functor::ArgMin<GPUDevice, T, int64>; \
  template struct functor::ArgMax<GPUDevice, T, int32>; \
  template struct functor::ArgMin<GPUDevice, T, int32>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPEC);

// (pin) for ArgMax2D
// define cuda kernel

template<typename TD, typename TI>
__device__ void argmax2d_thread_reduce(TD* data, TD* shared_data, TI* shared_index, TD lowest, int bsize, int hsize) {
  if (blockIdx.x > bsize) return;

  TD * batch_data = data + blockIdx.x * hsize;
  TD  thread_local_max_data = lowest;
  TI  thread_local_max_idx = 0;

  for (int i = threadIdx.x ; i < hsize; i += blockDim.x) {
    if (batch_data[i] > thread_local_max_data) {
      thread_local_max_data = batch_data[i];
      thread_local_max_idx = i;
    }
  }

  shared_data[threadIdx.x] = thread_local_max_data;
  shared_index[threadIdx.x] = thread_local_max_idx;
}

template<typename TD, typename TI>
__device__ void argmax2d_block_reduce(TD* shared_data, TI* shared_index) {
  int left, right;
  int threads = blockDim.x / 2;
  for (int stride = 1; stride < blockDim.x; stride *= 2, threads /=2 ) {
    if (threadIdx.x < threads) {
      left = threadIdx.x * (stride * 2);
      right = left + stride;
      if (shared_data[left] < shared_data[right]) {
        shared_data[left] = shared_data[right];
        shared_index[left] = shared_index[right];
      }
    }
    __syncthreads();
  }
}

template <typename TD, typename TI>
__global__ void argmax2d_kernel(TD* data, TI* index_out, TD lowest, int bsize, int hsize) {
  // shared memory
  extern __shared__ int s[];
  TD* shared_data = reinterpret_cast<TD*>(s);
  TI* shared_index = reinterpret_cast<TI*>(shared_data + blockDim.x);

  argmax2d_thread_reduce(data, shared_data, shared_index, lowest, bsize, hsize);
  __syncthreads();
  // for each block do block reduce
  argmax2d_block_reduce(shared_data, shared_index);

  // write to global memory
  if (threadIdx.x == 0) {
    index_out[blockIdx.x] = shared_index[0];
  }
}

// (pin) define GPU implementation that launch CUDA kernel.
namespace functor { 
template <typename T>
struct ArgMax2DFunctor<GPUDevice, T>{ 

  void operator() (const GPUDevice &d, const T* in, int* out, T lowest, int bsize, int hsize) { 
    // launch the CUDA kernel
     int numBlocks = bsize;
     int threadsPerBlock = 512;
     size_t dev_sm_bytes = threadsPerBlock * (sizeof(T) + sizeof(int));
     argmax2d_kernel<T, int> <<<numBlocks, threadsPerBlock, dev_sm_bytes, d.stream()>>>(const_cast<T*>(in), out,
                 lowest, bsize, hsize);
  }
};

#define DEFINE_ARGMAX2D_GPU_SPEC(T)                       \
  template struct ArgMax2DFunctor<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_ARGMAX2D_GPU_SPEC);

} // namespace functor




}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
