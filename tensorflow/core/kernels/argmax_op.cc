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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/argmax_op.h"

#include <memory>
#include <limits>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Tout, typename ArgFunctor>
class ArgOp : public OpKernel {
 public:
  explicit ArgOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& dimension = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(dimension.shape()),
                errors::InvalidArgument(
                    "dim must be a scalar, but received tensor of shape: ",
                    dimension.shape().DebugString()));

    const int32 dim = internal::SubtleMustCopy(dimension.scalar<int32>()());
    const int input_dims = input.dims();

    int axis = dim < 0 ? dim + input_dims : dim;

    OP_REQUIRES(context, axis >= 0 && axis < input_dims,
                errors::InvalidArgument("Expected dimension in the range [",
                                        -input_dims, ", ", input_dims,
                                        "), but got ", dim));
    OP_REQUIRES(
        context, input.dim_size(axis) > 0,
        errors::InvalidArgument("Reduction axis ", dim, " is empty in shape ",
                                input.shape().DebugString()));

    TensorShape output_shape;
    const TensorShape& input_shape = input.shape();
    for (int d = 0; d < input_dims - 1; ++d) {
      output_shape.AddDim(input_shape.dim_size((d < axis) ? d : d + 1));
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

#define HANDLE_DIM(NDIM)                                        \
  case NDIM:                                                    \
    ArgFunctor::Reduce##NDIM(context->eigen_device<Device>(),   \
                             input.tensor<T, NDIM>(), axis,     \
                             output->tensor<Tout, NDIM - 1>()); \
    break;

    switch (input_dims) {
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);

      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "ArgOp : Unhandled input dimensions: ", input_dims));
    }
  }
#undef HANDLE_DIM

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ArgOp);
};

template <typename Device, typename T, typename Tout>
class ArgMaxOp
    : public ArgOp<Device, T, Tout, functor::ArgMax<Device, T, Tout> > {
 public:
  explicit ArgMaxOp(OpKernelConstruction* context)
      : ArgOp<Device, T, Tout, functor::ArgMax<Device, T, Tout> >(context) {}
};

template <typename Device, typename T, typename Tout>
class ArgMinOp
    : public ArgOp<Device, T, Tout, functor::ArgMin<Device, T, Tout> > {
 public:
  explicit ArgMinOp(OpKernelConstruction* context)
      : ArgOp<Device, T, Tout, functor::ArgMin<Device, T, Tout> >(context) {}
};


namespace functor { 
/*
 * (pin) ArgMax2DOp
 */
template <typename T>
struct ArgMax2DFunctor<CPUDevice, T>{ 
  void operator() (const CPUDevice& d, 
                   const T* input, 
                   int* output,
                   T lowest,
                   int bsize,
                   int hsize) { 
    LOG(FATAL) << "Not Implemented.";
  }
};

} // namespace functor

template <typename Device, typename T>
class ArgMax2DOp : public OpKernel {
 public : 
  explicit ArgMax2DOp(OpKernelConstruction* context) : OpKernel(context) { } 

  void Compute(OpKernelContext* context) override { 
    const Tensor& input = context->input(0);
    const int input_dims = input.dims();
    OP_REQUIRES(context, input_dims == 2,
                errors::InvalidArgument("Expected dimension 2, but got ", input_dims));
    int axis = 1;
    OP_REQUIRES(
        context, input.dim_size(axis) > 0,
        errors::InvalidArgument("Reduction axis ", axis, " is empty in shape ",
                                input.shape().DebugString()));
    TensorShape output_shape;
    const TensorShape& input_shape = input.shape();
    output_shape.AddDim(input_shape.dim_size(0)); // assume first dim is batch dim.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    /*
    if (std::is_same<Device, GPUDevice>::value) { 
      LOG(INFO) << " FUCK It IS GPU !!! FUCK YOU .";
    } else if (std::is_same<Device, CPUDevice>::value) { 
      LOG(INFO) << " FUCK !!!!!!!!!!!! CPU .";
    } else { 
      LOG(FATAL) << " FUCK NON!";
    }
    */

    // compute
    int bsize = input_shape.dim_size(0);
    int hsize = input_shape.dim_size(1);
    functor::ArgMax2DFunctor<Device, T>()(
      context->eigen_device<Device>(),
      input.flat<T>().data(),
      output->flat<int>().data(),
      std::numeric_limits<T>::min(),
      bsize,
      hsize);
  }
};

#define REGISTER_ARGMAX2D_CPU(type) \
  REGISTER_KERNEL_BUILDER(Name("ArgMax2D") \
                          .Device(DEVICE_CPU) \
                          .TypeConstraint<type>("T"), \
                        ArgMax2DOp<CPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_ARGMAX2D_CPU);





#define REGISTER_ARGMAX(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                            \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64>("output_type") \
                              .HostMemory("dimension"),             \
                          ArgMaxOp<CPUDevice, type, int64>);        \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                            \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64>("output_type") \
                              .HostMemory("dimension"),             \
                          ArgMinOp<CPUDevice, type, int64>);        \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                            \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int32>("output_type") \
                              .HostMemory("dimension"),             \
                          ArgMaxOp<CPUDevice, type, int32>);        \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                            \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int32>("output_type") \
                              .HostMemory("dimension"),             \
                          ArgMinOp<CPUDevice, type, int32>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_ARGMAX);

#if GOOGLE_CUDA

// Forward declarations of the functor specializations for GPU.
namespace functor {

#define DECLARE_GPU_SPEC(T, Tout, Dims)                                       \
  template <>                                                                 \
  void ArgMax<GPUDevice, T, Tout>::Reduce##Dims(                              \
      const GPUDevice& d, typename TTypes<T, Dims>::ConstTensor input,        \
      const int32 dimension, typename TTypes<Tout, Dims - 1>::Tensor output); \
  template <>                                                                 \
  void ArgMin<GPUDevice, T, Tout>::Reduce##Dims(                              \
      const GPUDevice& d, typename TTypes<T, Dims>::ConstTensor input,        \
      const int32 dimension, typename TTypes<Tout, Dims - 1>::Tensor output);

#define DECLARE_GPU_SPECS(T)     \
  DECLARE_GPU_SPEC(T, int64, 1); \
  DECLARE_GPU_SPEC(T, int64, 2); \
  DECLARE_GPU_SPEC(T, int64, 3); \
  DECLARE_GPU_SPEC(T, int64, 4); \
  DECLARE_GPU_SPEC(T, int64, 5); \
  DECLARE_GPU_SPEC(T, int32, 1); \
  DECLARE_GPU_SPEC(T, int32, 2); \
  DECLARE_GPU_SPEC(T, int32, 3); \
  DECLARE_GPU_SPEC(T, int32, 4); \
  DECLARE_GPU_SPEC(T, int32, 5);

#define DECLARE_GPU_CLASS(T)                          \
  extern template struct ArgMax<GPUDevice, T, int64>; \
  extern template struct ArgMin<GPUDevice, T, int64>; \
  extern template struct ArgMax<GPUDevice, T, int32>; \
  extern template struct ArgMin<GPUDevice, T, int32>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_CLASS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_CLASS


// (pin)  forward declare gpu functor
#define DECLARE_ARGMAX2D_FUNCTOR_GPU_SPEC(T)        \
  template <>                                       \
  void ArgMax2DFunctor<GPUDevice, T>::operator()    \
    (const GPUDevice &d, const T* in, int* out, T lowest, int bsize, int hsize) ;

  TF_CALL_GPU_NUMBER_TYPES(DECLARE_ARGMAX2D_FUNCTOR_GPU_SPEC);

  #define DECLARE_ARGMAX2D_FUNCTOR_GPU_CLASS(T) \
    extern template struct ArgMax2DFunctor<GPUDevice, T>;
  
    TF_CALL_GPU_NUMBER_TYPES(DECLARE_ARGMAX2D_FUNCTOR_GPU_CLASS);
  #undef DECLARE_ARGMAX2D_FUNCTOR_GPU_SPEC
#undef DECLARE_ARGMAX2D_FUNCTOR_GPU_CLASS


}  // namespace functor

// Registration of the GPU implementations.
// (pin) 
#define REGISTER_ARGMAX2D_GPU(type) \
  REGISTER_KERNEL_BUILDER(Name("ArgMax2D") \
                          .Device(DEVICE_GPU) \
                          .TypeConstraint<type>("T"), \
                        ArgMax2DOp<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_ARGMAX2D_GPU);
#undef REGISTER_ARGMAX2D_GPU


// Registration of the GPU implementations.
#define REGISTER_ARGMAX_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                            \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64>("output_type") \
                              .TypeConstraint<int32>("Tidx")        \
                              .HostMemory("dimension"),             \
                          ArgMaxOp<GPUDevice, type, int64>);        \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                            \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int64>("output_type") \
                              .TypeConstraint<int32>("Tidx")        \
                              .HostMemory("dimension"),             \
                          ArgMinOp<GPUDevice, type, int64>);        \
  REGISTER_KERNEL_BUILDER(Name("ArgMax")                            \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int32>("output_type") \
                              .TypeConstraint<int32>("Tidx")        \
                              .HostMemory("dimension"),             \
                          ArgMaxOp<GPUDevice, type, int32>);        \
  REGISTER_KERNEL_BUILDER(Name("ArgMin")                            \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<type>("T")            \
                              .TypeConstraint<int32>("output_type") \
                              .TypeConstraint<int32>("Tidx")        \
                              .HostMemory("dimension"),             \
                          ArgMinOp<GPUDevice, type, int32>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_ARGMAX_GPU);

#undef REGISTER_ARGMAX_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
