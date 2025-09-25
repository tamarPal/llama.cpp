//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "conv.hpp"

static  void conv_transpose_1d_kernel(
        const int s0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2,
        const int src1_ne0, const int dst_ne0,
        const float * src0, const float * src1,  float * dst,
        const sycl::nd_item<3> &item_ct1) {
    int global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;

    float accumulator = 0;

    for (int c = 0; c < src0_ne2; c++) {
        int idx = global_index % dst_ne0;

        int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        int input_offset = src1_ne0 * c;

        for (int i = 0; i < src1_ne0; i++) {
            if (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                continue;
            }
            int weight_idx = idx - i*s0;

            float kernel_weight = src0[kernel_offset + weight_idx];
            float input_value =  src1[input_offset+i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
}

static void conv_transpose_1d_f32_f32_sycl(
    const int s0, const int output_size,
    const int src0_ne0, const int src0_ne1, const int src0_ne2,
    const int src1_ne0, const int dst_ne0,
    const float *src0, const float *src1, float *dst,
    const queue_ptr& stream) {

    const int num_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    const sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, 1, num_blocks);
    sycl_parallel_for(stream, sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
        conv_transpose_1d_kernel(s0, output_size, src0_ne0, src0_ne1, src0_ne2, src1_ne0, dst_ne0, src0, src1, dst,
                                 item_ct1);
    });
}

void ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = opts[0];

    const int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

// Conv2D kernel implementation
static void conv_2d_kernel(
        const int h, const int w, const int kh, const int kw,
        const int s0, const int s1, const int p0, const int p1, const int d0, const int d1,
        const int src0_ne0, const int src0_ne1, const int src0_ne2,
        const int src1_ne0, const int src1_ne1, const int src1_ne2,
        const int dst_ne0, const int dst_ne1, const int dst_ne2,
        const float *src0, const float *src1, float *dst,
        const sycl::nd_item<3> &item_ct1) {
        
    // Get thread position
    int out_h = item_ct1.get_global_id(0);
    int out_w = item_ct1.get_global_id(1);
    int out_c = item_ct1.get_global_id(2);
    
    if (out_h >= dst_ne1 || out_w >= dst_ne0 || out_c >= dst_ne2) {
        return;
    }
    
    float sum = 0.0f;
    
    // Convolution computation
    for (int ic = 0; ic < src1_ne2; ic++) {
        for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
            for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
                int in_h = out_h * s1 - p1 + kh_idx * d1;
                int in_w = out_w * s0 - p0 + kw_idx * d0;
                
                if (in_h >= 0 && in_h < src1_ne1 && in_w >= 0 && in_w < src1_ne0) {
                    int kernel_idx = (out_c * src1_ne2 + ic) * kh * kw + kh_idx * kw + kw_idx;
                    int input_idx = ic * src1_ne1 * src1_ne0 + in_h * src1_ne0 + in_w;
                    
                    sum += src0[kernel_idx] * src1[input_idx];
                }
            }
        }
    }
    
    int out_idx = out_c * dst_ne1 * dst_ne0 + out_h * dst_ne0 + out_w;
    dst[out_idx] = sum;
}

static void conv_2d_f32_sycl(
    const int h, const int w, const int kh, const int kw,
    const int s0, const int s1, const int p0, const int p1, const int d0, const int d1,
    const int src0_ne0, const int src0_ne1, const int src0_ne2,
    const int src1_ne0, const int src1_ne1, const int src1_ne2,
    const int dst_ne0, const int dst_ne1, const int dst_ne2,
    const float *src0, const float *src1, float *dst,
    const queue_ptr& stream) {
    
    fprintf(stderr, "🔥 SYCL Conv2D Kernel: Launching SYCL kernels on GPU!\n");
    
    const int num_threads = 256;
    const sycl::range<3> block_size(8, 8, 4);
    const sycl::range<3> grid_size(
        (dst_ne1 + block_size[0] - 1) / block_size[0],
        (dst_ne0 + block_size[1] - 1) / block_size[1],
        (dst_ne2 + block_size[2] - 1) / block_size[2]
    );
    
    fprintf(stderr, "🔥 SYCL Conv2D Kernel: Grid size: %lux%lux%lu, Block size: %lux%lux%lu\n",
            grid_size[0], grid_size[1], grid_size[2], 
            block_size[0], block_size[1], block_size[2]);
    
    sycl_parallel_for(stream, sycl::nd_range<3>(grid_size * block_size, block_size), 
        [=](sycl::nd_item<3> item_ct1) {
            conv_2d_kernel(h, w, kh, kw, s0, s1, p0, p1, d0, d1,
                          src0_ne0, src0_ne1, src0_ne2,
                          src1_ne0, src1_ne1, src1_ne2,
                          dst_ne0, dst_ne1, dst_ne2,
                          src0, src1, dst, item_ct1);
        });
    
    fprintf(stderr, "🔥 SYCL Conv2D Kernel: SYCL parallel_for completed!\n");
}

void ggml_sycl_op_conv_2d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    
    // Debug prints to track execution
    fprintf(stderr, "🔥 SYCL Conv2D: Starting Conv2D operation!\n");
    fprintf(stderr, "🔥 SYCL Conv2D: Function called successfully\n");
    
    const ggml_tensor *src0 = dst->src[0]; // kernel
    const ggml_tensor *src1 = dst->src[1]; // input
    
    const float *src0_d = (const float *)src0->data;
    const float *src1_d = (const float *)src1->data;
    float *dst_d = (float *)dst->data;
    
    dpct::queue_ptr stream = ctx.stream();
    
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    
    const int32_t *opts = (const int32_t *)dst->op_params;
    const int s0 = opts[0]; // stride w
    const int s1 = opts[1]; // stride h  
    const int p0 = opts[2]; // pad w
    const int p1 = opts[3]; // pad h
    const int d0 = opts[4]; // dilation w
    const int d1 = opts[5]; // dilation h
    
    const int kw = src0->ne[0]; // kernel width
    const int kh = src0->ne[1]; // kernel height
    const int w = src1->ne[0];  // input width
    const int h = src1->ne[1];  // input height
    
    // Debug: print tensor dimensions
    fprintf(stderr, "🔥 SYCL Conv2D: Input dimensions: %dx%dx%d\n", w, h, (int)src1->ne[2]);
    fprintf(stderr, "🔥 SYCL Conv2D: Kernel dimensions: %dx%dx%d\n", kw, kh, (int)src0->ne[2]);
    fprintf(stderr, "🔥 SYCL Conv2D: Output dimensions: %dx%dx%d\n", (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2]);
    fprintf(stderr, "🔥 SYCL Conv2D: Parameters: stride=(%d,%d), pad=(%d,%d), dilation=(%d,%d)\n", 
            s0, s1, p0, p1, d0, d1);
    
    conv_2d_f32_sycl(
        h, w, kh, kw, s0, s1, p0, p1, d0, d1,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], src1->ne[1], src1->ne[2],
        dst->ne[0], dst->ne[1], dst->ne[2],
        src0_d, src1_d, dst_d, stream);
    
    fprintf(stderr, "🔥 SYCL Conv2D: Conv2D operation completed successfully!\n");
}


