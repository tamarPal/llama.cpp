#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
#include "ggml.h"

// אין תלות ב-helpers פנימיים; פותחים תור GPU מקומי
using namespace sycl;

static void kernel_roll_impl(queue &q,
                             const ggml_tensor *src,
                             ggml_tensor *dst,
                             int axis, int shift) {
    // בדיקות בסיס
    if (!src || !dst) throw std::runtime_error("null tensor");
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        throw std::runtime_error("only F32 supported in SYCL roll (for test)");
    if (axis < 0 || axis > 3) throw std::runtime_error("axis out of range");

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    // התאמת ממדים למקור
    if (ne0 != src->ne[0] || ne1 != src->ne[1] || ne2 != src->ne[2] || ne3 != src->ne[3])
        throw std::runtime_error("src/dst shape mismatch");

    const int64_t len = dst->ne[axis];
    if (len <= 0) throw std::runtime_error("len <= 0");

    // shift חיובי בתחום [0, len)
    const int64_t sh = ((int64_t)shift % len + len) % len;

    const size_t total_elems = (size_t)ne0 * (size_t)ne1 * (size_t)ne2 * (size_t)ne3;
    const size_t total_bytes = total_elems * sizeof(float);

    // מצביעי ה-host
    const float *h_src = (const float*) src->data;
    float *h_dst = (float*) dst->data;
    if (!h_src || !h_dst) throw std::runtime_error("null data pointers");

    // USM משותף — קריא/כתיב מה-CPU ומה-GPU
    float *usm_src = (float*) malloc_shared(total_bytes, q);
    float *usm_dst = (float*) malloc_shared(total_bytes, q);
    if (!usm_src || !usm_dst) throw std::runtime_error("malloc_shared failed");

    // העתקות host<->usm
    std::memcpy(usm_src, h_src, total_bytes);
    std::memset(usm_dst, 0, total_bytes);

    // kernel: פריסה פשוטה (i3,i2,i1) לולאה פנימית על i0
    q.submit([&](handler &h) {
        range<3> r((size_t)ne3, (size_t)ne2, (size_t)ne1);
        h.parallel_for(r, [=](id<3> idx) {
            int64_t i3 = (int64_t)idx[0];
            int64_t i2 = (int64_t)idx[1];
            int64_t i1 = (int64_t)idx[2];
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                int64_t s0=i0, s1=i1, s2=i2, s3=i3;
                if (axis==0) s0 = (i0 - sh + ne0) % ne0;
                if (axis==1) s1 = (i1 - sh + ne1) % ne1;
                if (axis==2) s2 = (i2 - sh + ne2) % ne2;
                if (axis==3) s3 = (i3 - sh + ne3) % ne3;

                size_t src_idx = (size_t)(((s3*ne2 + s2)*ne1 + s1)*ne0 + s0);
                size_t dst_idx = (size_t)(((i3*ne2 + i2)*ne1 + i1)*ne0 + i0);

                // שמירה בטוחה
                usm_dst[dst_idx] = usm_src[src_idx];
            }
        });
    }).wait();

    // החזרת תוצאה ל-host
    std::memcpy(h_dst, usm_dst, total_bytes);

    sycl::free(usm_src, q);
    sycl::free(usm_dst, q);
}

// API יציב ל-backend שלנו
extern "C" void ggml_sycl_roll(ggml_tensor * dst,
                               const ggml_tensor * src,
                               int axis, int shift) {
    try {
        queue q{ gpu_selector_v };
        kernel_roll_impl(q, src, dst, axis, shift);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        throw;
    }
}

// שמירה על התאימות לטסטים/קריאות ישנות
extern "C" void ggml_sycl_roll_probe(ggml_tensor * src,
                                     ggml_tensor * dst,
                                     int axis,
                                     long long shift) {
    ggml_sycl_roll(dst, src, axis, (int)shift);
}
