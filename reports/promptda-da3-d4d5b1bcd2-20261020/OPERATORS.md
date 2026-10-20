# PyTorch operator detail

每个模型另执行一次完整请求，用 PyTorch profiler 记录 CPU/CUDA 活动与 input shapes，不记录 tensor、stack 或 memory trace。下面的算子归因与底层 kernel 是重叠视图，不相加；self device 指 profiler 的归因统计，不等同于请求 wall。完整未截断名称见 operator_timings.csv。

## da3

| 算子 / kernel | 次数 | self CPU / ms | self device / ms | input shapes |
| --- | --- | --- | --- | --- |
| void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t, Flash_kernel_trai | 64 | 0.000 | 277.140 | [] |
| ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_tn | 216 | 0.000 | 250.076 | [] |
| aten::_flash_attention_forward | 14 | 0.294 | 237.124 | [[1, 19460, 24, 64], [1, 19460, 24, 64], [1, 19460, 24, 64], [], [], [], [], [], [], [], [], [], [], [], []] |
| Command Buffer Full | 326 | 211.584 | 133.254 | [] |
| xformers::dual_gemm_silu_identity_mul | 40 | 1.927 | 131.957 | [[19460, 1536], [4096, 1536], [4096], [4096, 1536], [4096]] |
| _ZN7cutlass6KernelINS_4gemm6kernel8DualGemmINS1_11threadblock17DualMmaMultistageINS1_9GemmShapeILi128ELi64ELi32EEENS_9transform11t | 40 | 0.000 | 131.957 | [] |
| sm86_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x16_stage3_warpsize2x2x1_g1_tensor16x8x8_execute_ke | 95 | 0.000 | 117.667 | [] |
| aten::addmm | 40 | 1.378 | 75.069 | [[4608], [19460, 1536], [1536, 4608], [], []] |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CU | 343 | 0.000 | 73.622 | [] |
| void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float>(int, float, float const*, float const*, float c | 215 | 0.000 | 71.687 | [] |
| aten::addmm | 40 | 1.481 | 69.055 | [[1536], [19460, 4096], [4096, 1536], [], []] |
| aten::cudnn_convolution | 28 | 1.045 | 64.574 | [[8, 256, 108, 144], [256, 256, 3, 3], [], [], [], [], [], [], []] |
| aten::cudnn_convolution | 10 | 0.361 | 50.676 | [[8, 256, 216, 288], [128, 256, 3, 3], [], [], [], [], [], [], []] |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIter | 335 | 0.000 | 45.137 | [] |
| sm86_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x32_stage3_warpsize2x2x1_g1_tensor16x8x8_alignc4_ex | 28 | 0.000 | 42.701 | [] |
| void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1 | 700 | 0.000 | 41.229 | [] |
| aten::cat | 265 | 14.643 | 37.813 | [[], []] |
| void cudnn::engines_precompiled::convertTensor_kernel<float, float, float, (cudnnKernelDataType_t)2>(float, float const*, float, f | 348 | 0.000 | 31.171 | [] |
| aten::addmm | 24 | 0.630 | 26.612 | [[1024], [19460, 4096], [4096, 1024], [], []] |
| aten::addmm | 40 | 1.698 | 26.349 | [[1536], [19460, 1536], [1536, 1536], [], []] |
| aten::addmm | 24 | 0.718 | 26.175 | [[4096], [19460, 1024], [1024, 4096], [], []] |
| aten::_flash_attention_forward | 26 | 0.553 | 24.839 | [[20, 973, 24, 64], [20, 973, 24, 64], [20, 973, 24, 64], [], [], [], [], [], [], [], [], [], [], [], []] |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at:: | 282 | 0.000 | 24.560 | [] |
| void cutlass__5x_cudnn::Kernel<cutlass_tensorop_s1688fprop_optimized_tf32_128x128_16x4_nhwc_align4>(cutlass_tensorop_s1688fprop_op | 20 | 0.000 | 23.099 | [] |
| aten::cudnn_convolution | 6 | 0.237 | 21.430 | [[8, 128, 378, 504], [32, 128, 3, 3], [], [], [], [], [], [], []] |
| aten::addmm | 24 | 0.872 | 19.921 | [[3072], [19460, 1024], [1024, 3072], [], []] |
| aten::native_layer_norm | 28 | 0.505 | 19.819 | [[1, 24, 19460, 64], [], [64], [64], []] |
| aten::add | 52 | 0.562 | 19.772 | [[20, 973, 1536], [20, 973, 1536], []] |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native: | 128 | 0.000 | 19.428 | [] |
| void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 4, 64 | 114 | 0.000 | 19.375 | [] |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorItera | 263 | 0.000 | 18.989 | [] |
| aten::native_layer_norm | 26 | 0.485 | 18.335 | [[20, 24, 973, 64], [], [64], [64], []] |
| aten::cudnn_convolution | 4 | 0.143 | 17.745 | [[8, 128, 216, 288], [256, 128, 3, 3], [], [], [], [], [], [], []] |
| void at::native::(anonymous namespace)::upsample_bicubic2d_out_frame<float, float>(int, float, float, bool, at::GenericPackedTenso | 2 | 0.000 | 17.308 | [] |
| aten::copy_ | 84 | 0.696 | 17.099 | [[1, 24, 19460, 64], [1, 24, 19460, 64], []] |

## promptda

| 算子 / kernel | 次数 | self CPU / ms | self device / ms | input shapes |
| --- | --- | --- | --- | --- |
| ampere_fp16_s16816gemm_fp16_128x64_ldg8_relu_f2f_tn | 960 | 0.000 | 330.669 | [] |
| flash_attn_3::fwd | 240 | 5.017 | 216.636 | [[2, 3889, 16, 64], [2, 3889, 16, 64], [2, 3889, 16, 64], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []] |
| void cutlass::device_kernel<flash::enable_sm80_to_sm89<flash::FlashAttnFwdSm80<flash::CollectiveMainloopFwdSm80<4, 1, false, cute: | 240 | 0.000 | 216.636 | [] |
| aten::addmm | 240 | 8.483 | 110.341 | [[4096], [7778, 1024], [1024, 4096], [], []] |
| aten::addmm | 240 | 7.544 | 108.943 | [[1024], [7778, 4096], [4096, 1024], [], []] |
| aten::addmm | 240 | 8.414 | 81.377 | [[3072], [7778, 1024], [1024, 3072], [], []] |
| sm86_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x128x64_stage2_warpsize4x2x1_g1_tensor16x8x16_execute_ker | 120 | 0.000 | 79.867 | [] |
| aten::cudnn_convolution | 70 | 2.444 | 78.021 | [[2, 256, 216, 288], [256, 256, 3, 3], [], [], [], [], [], [], []] |
| void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1} | 3360 | 0.000 | 59.192 | [] |
| Memcpy HtoD (Pageable -> Device) | 20 | 0.000 | 41.552 | [] |
| aten::copy_ | 20 | 0.296 | 41.276 | [[2, 3, 756, 1008], [2, 3, 756, 1008], []] |
| sm86_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize128x128x32_stage3_warpsize2x2x1_g1_tensor16x8x16_execute_ker | 30 | 0.000 | 40.132 | [] |
| aten::upsample_bicubic2d | 10 | 0.279 | 39.027 | [[1, 1024, 37, 37], [], [], []] |
| void at::native::(anonymous namespace)::upsample_bicubic2d_out_frame<float, float>(int, float, float, bool, at::GenericPackedTenso | 10 | 0.000 | 39.027 | [] |
| void at::native::(anonymous namespace)::upsample_bilinear2d_nhwc_out_frame<float, float>(float, float, bool, int, int, int, int, i | 50 | 0.000 | 39.014 | [] |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CU | 500 | 0.000 | 30.910 | [] |
| aten::add | 480 | 5.333 | 30.889 | [[2, 3889, 1024], [2, 3889, 1024], []] |
| aten::addmm | 240 | 9.156 | 30.694 | [[1024], [7778, 1024], [1024, 1024], [], []] |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::Half> >(at::Tensor | 450 | 0.000 | 29.869 | [] |
| aten::gelu | 240 | 3.116 | 28.172 | [[2, 3889, 4096], []] |
| void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{ | 240 | 0.000 | 28.172 | [] |
| aten::cudnn_convolution | 60 | 2.102 | 21.383 | [[2, 256, 108, 144], [256, 256, 3, 3], [], [], [], [], [], [], []] |
| aten::upsample_bilinear2d | 10 | 0.103 | 21.146 | [[2, 128, 432, 576], [], [], [], []] |
| aten::cudnn_convolution | 10 | 0.344 | 19.770 | [[2, 256, 432, 576], [128, 256, 3, 3], [], [], [], [], [], [], []] |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native: | 490 | 0.000 | 17.332 | [] |
| aten::mul | 480 | 8.120 | 17.261 | [[2, 3889, 1024], [1024]] |
| aten::native_layer_norm | 520 | 9.753 | 16.198 | [[2, 3889, 1024], [], [1024], [1024], []] |
| void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float>(int, float, float const*, float const*, float c | 520 | 0.000 | 16.198 | [] |
| aten::upsample_bilinear2d | 10 | 0.106 | 13.567 | [[2, 256, 216, 288], [], [], [], []] |
| aten::copy_ | 10 | 0.048 | 12.179 | [[2, 128, 756, 1008], [2, 128, 756, 1008], []] |
| aten::copy_ | 480 | 2.396 | 11.665 | [[2, 3889, 1024], [2, 3889, 1024], []] |
| aten::copy_ | 10 | 0.125 | 11.305 | [[2, 756, 1008], [2, 756, 1008], []] |
| Memcpy DtoH (Device -> Pageable) | 10 | 0.000 | 11.305 | [] |
| aten::add_ | 90 | 0.761 | 10.633 | [[2, 256, 216, 288], [1, 256, 1, 1], []] |
| aten::add | 40 | 0.456 | 9.913 | [[2, 256, 216, 288], [2, 256, 216, 288], []] |
