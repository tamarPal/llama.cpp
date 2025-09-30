#include <sycl/sycl.hpp>
#include "ggml.h"
#include "common.hpp"

// קרנל כללי ל-f32 / f16
template<typename T>
static void kernel_roll_impl(
    sycl::queue &q,
    const T* src, T* dst,
    const int64_t *ne_in,
    const size_t  *nb_in,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3) {

    // נעתיק לערכים לוקאליים כדי לא לקרוא מהוסט בתוך הקרנל
    const int64_t n0 = ne_in[0], n1 = ne_in[1], n2 = ne_in[2], n3 = ne_in[3];
    const size_t  nb0 = nb_in[0], nb1 = nb_in[1], nb2 = nb_in[2], nb3 = nb_in[3];

    auto norm_shift = [](int64_t s, int64_t n) -> int64_t {
        if (n <= 0) return 0;
        int64_t r = s % n; if (r < 0) r += n; return r;
    };

    const int64_t rs0 = norm_shift(s0, n0);
    const int64_t rs1 = norm_shift(s1, n1);
    const int64_t rs2 = norm_shift(s2, n2);
    const int64_t rs3 = norm_shift(s3, n3);

    const int64_t total = (n0>0 && n1>0 && n2>0 && n3>0) ? (n0 * n1 * n2 * n3) : 0;
    if (total == 0) return;

    q.parallel_for(sycl::range<1>(static_cast<size_t>(total)), [=](sycl::id<1> tid) {
        int64_t i = static_cast<int64_t>(tid[0]);

        int64_t i0 = i % n0;      i /= n0;
        int64_t i1 = i % n1;      i /= n1;
        int64_t i2 = i % n2;      i /= n2;
        int64_t i3 = i;

        const int64_t j0 = (i0 - rs0 + n0) % n0;
        const int64_t j1 = (i1 - rs1 + n1) % n1;
        const int64_t j2 = (i2 - rs2 + n2) % n2;
        const int64_t j3 = (i3 - rs3 + n3) % n3;

        const size_t off_src = (size_t)j0 * nb0 + (size_t)j1 * nb1 + (size_t)j2 * nb2 + (size_t)j3 * nb3;
        const size_t off_dst = (size_t)i0 * nb0 + (size_t)i1 * nb1 + (size_t)i2 * nb2 + (size_t)i3 * nb3;

        const T* psrc = (const T*)((const char*)src + off_src);
        T*       pdst = (T*)      ((      char*)dst + off_dst);
        *pdst = *psrc;
    }).wait();
}

// חתימת ה-forward כפי שנקראת מ-ggml-sycl.cpp
bool ggml_sycl_compute_forward_roll(ggml_backend_sycl_context &ctx, const ggml_tensor *src, ggml_tensor *dst) {
    GGML_ASSERT(src != nullptr && dst != nullptr);

    // 4 שיפטים (int32) ב-op_params: shift0..shift3
    const int32_t *op_params = (const int32_t *) dst->op_params;
    const int64_t s0 = (int64_t)op_params[0];
    const int64_t s1 = (int64_t)op_params[1];
    const int64_t s2 = (int64_t)op_params[2];
    const int64_t s3 = (int64_t)op_params[3];

    const int64_t *ne = dst->ne;
    const size_t  *nb = dst->nb;

    const void * src_data = src->data;
    void       * dst_data = dst->data;

    // הבאת queue באופן קנוני ב-SYCL backend של GGML:
    auto stream = ctx.stream();
    sycl::queue &q = *stream;

    switch (ggml_element_size(dst)) {
    case sizeof(float):
        kernel_roll_impl<float>(q,
            (const float*)src_data, (float*)dst_data,
            ne, nb, s0, s1, s2, s3);
        break;
    case sizeof(ggml_fp16_t):
        kernel_roll_impl<ggml_fp16_t>(q,
            (const ggml_fp16_t*)src_data, (ggml_fp16_t*)dst_data,
            ne, nb, s0, s1, s2, s3);
        break;
    default:
        GGML_ABORT("roll: unsupported type on SYCL");
        return false;
    }

    return true;
}
