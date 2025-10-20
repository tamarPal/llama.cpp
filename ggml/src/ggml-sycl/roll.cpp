#include "roll.hpp"
#include "common.hpp"

// Normalize negative/large shifts to valid range [0, n)
static inline int64_t norm_shift(int s, int64_t n) {
    if (n <= 0) return 0;
    int64_t ss = (int64_t)s % n;
    if (ss < 0) ss += n;
    return ss;
}

// General 4D kernel, supports strides (in elements, not bytes) - Template version for F32/F16
template<typename T>
static sycl::event kernel_roll_4d(
    sycl::queue &q,
    const T *src_d, T *dst_d,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    // source/dest strides in elements
    int64_t nb0s, int64_t nb1s, int64_t nb2s, int64_t nb3s,
    int64_t nb0d, int64_t nb1d, int64_t nb2d, int64_t nb3d,
    // normalized shifts
    int64_t sh0, int64_t sh1, int64_t sh2, int64_t sh3
) {
    const sycl::range<3> r{ (size_t)ne3, (size_t)ne2, (size_t)ne1 };

    return q.submit([&](sycl::handler &h) {
        h.parallel_for(r, [=](sycl::id<3> idx) {
            const int64_t i3 = (int64_t)idx[0];
            const int64_t i2 = (int64_t)idx[1];
            const int64_t i1 = (int64_t)idx[2];

            const int64_t si3 = (i3 - sh3 + ne3) % ne3;
            const int64_t si2 = (i2 - sh2 + ne2) % ne2;
            const int64_t si1 = (i1 - sh1 + ne1) % ne1;

            const int64_t base_dst = i1 * nb1d + i2 * nb2d + i3 * nb3d;
            const int64_t base_src = si1 * nb1s + si2 * nb2s + si3 * nb3s;

            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int64_t si0 = (i0 - sh0 + ne0) % ne0;
                const int64_t idx_dst = base_dst + i0 * nb0d;
                const int64_t idx_src = base_src + si0 * nb0s;
                dst_d[idx_dst] = src_d[idx_src];
            }
        });
    });
}

// Helper function to dispatch to correct template based on type
static sycl::event roll_dispatch_by_type(
    sycl::queue &q, ggml_type type,
    const void *src_d, void *dst_d,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    int64_t nb0s, int64_t nb1s, int64_t nb2s, int64_t nb3s,
    int64_t nb0d, int64_t nb1d, int64_t nb2d, int64_t nb3d,
    int64_t sh0, int64_t sh1, int64_t sh2, int64_t sh3
) {
    switch (type) {
        case GGML_TYPE_F32:
            return kernel_roll_4d<float>(q, 
                (const float*)src_d, (float*)dst_d,
                ne0, ne1, ne2, ne3, nb0s, nb1s, nb2s, nb3s,
                nb0d, nb1d, nb2d, nb3d, sh0, sh1, sh2, sh3);
        case GGML_TYPE_F16:
            return kernel_roll_4d<sycl::half>(q,
                (const sycl::half*)src_d, (sycl::half*)dst_d,
                ne0, ne1, ne2, ne3, nb0s, nb1s, nb2s, nb3s,
                nb0d, nb1d, nb2d, nb3d, sh0, sh1, sh2, sh3);
        default:
            GGML_ABORT("Unsupported type for ROLL operation");
    }
}

void ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    // Input validation
    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(dst->src[0] != nullptr);
    
    const ggml_tensor *src = dst->src[0];
    
    // Type validation - supports F32 and F16
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == src->type); // Types must match
    
    // Data pointer validation
    GGML_ASSERT(src->data != nullptr);
    GGML_ASSERT(dst->data != nullptr);

    // Extract shift parameters
    const int32_t *p = (const int32_t *) dst->op_params;
    
    // Dimension validation - ensure all dimensions are positive
    GGML_ASSERT(dst->ne[0] > 0 && dst->ne[1] > 0 && dst->ne[2] > 0 && dst->ne[3] > 0);
    GGML_ASSERT(src->ne[0] > 0 && src->ne[1] > 0 && src->ne[2] > 0 && src->ne[3] > 0);
    
    const int64_t sh0 = norm_shift(p[0], dst->ne[0]);
    const int64_t sh1 = norm_shift(p[1], dst->ne[1]);
    const int64_t sh2 = norm_shift(p[2], dst->ne[2]);
    const int64_t sh3 = norm_shift(p[3], dst->ne[3]);

    // shape validation
    GGML_ASSERT(dst->ne[0] == src->ne[0] &&
                dst->ne[1] == src->ne[1] &&
                dst->ne[2] == src->ne[2] &&
                dst->ne[3] == src->ne[3]);

    // Stride validation - ensure strides are properly aligned for the element type
    const size_t element_size = ggml_type_size(src->type);
    GGML_ASSERT(src->nb[0] % element_size == 0);
    GGML_ASSERT(src->nb[1] % element_size == 0);
    GGML_ASSERT(src->nb[2] % element_size == 0);
    GGML_ASSERT(src->nb[3] % element_size == 0);
    
    GGML_ASSERT(dst->nb[0] % element_size == 0);
    GGML_ASSERT(dst->nb[1] % element_size == 0);
    GGML_ASSERT(dst->nb[2] % element_size == 0);
    GGML_ASSERT(dst->nb[3] % element_size == 0);

    sycl::queue *q = ctx.stream();
    GGML_ASSERT(q != nullptr);

    // fast path - no shift: just memcpy
    if (sh0 == 0 && sh1 == 0 && sh2 == 0 && sh3 == 0) {
        (void) q->memcpy(dst->data, src->data, ggml_nbytes(dst)).wait();
        return;
    }

    // strides in elements (not bytes!)
    const int64_t nb0s = src->nb[0] / (int64_t)element_size;
    const int64_t nb1s = src->nb[1] / (int64_t)element_size;
    const int64_t nb2s = src->nb[2] / (int64_t)element_size;
    const int64_t nb3s = src->nb[3] / (int64_t)element_size;

    const int64_t nb0d = dst->nb[0] / (int64_t)element_size;
    const int64_t nb1d = dst->nb[1] / (int64_t)element_size;
    const int64_t nb2d = dst->nb[2] / (int64_t)element_size;
    const int64_t nb3d = dst->nb[3] / (int64_t)element_size;

    const void *src_d = src->data;
    void *dst_d = dst->data;

    // in-place: copy first to temp buffer to avoid overwriting source during write
    void *tmp_dev = nullptr;
    sycl::event ev_copy;

    if (dst->data == src->data) {
        const size_t nbytes = ggml_nbytes(src);
        tmp_dev = sycl::malloc_device(nbytes, *q);
        GGML_ASSERT(tmp_dev && "malloc_device failed");
        ev_copy = q->memcpy(tmp_dev, src_d, nbytes); // no wait here
        src_d = tmp_dev; // kernel will read from safe copy
    }

    sycl::event ev_kernel = roll_dispatch_by_type(
        *q, src->type,
        src_d, dst_d,
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        nb0s, nb1s, nb2s, nb3s,
        nb0d, nb1d, nb2d, nb3d,
        sh0, sh1, sh2, sh3
    );

    if (tmp_dev) {
        ev_copy.wait();
        ev_kernel.wait();
        sycl::free(tmp_dev, *q);
    } else {
        ev_kernel.wait();
    }
}