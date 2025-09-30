// tests/roll_sycl_sanity.cpp — pointer graph + two 1D rolls
#include <cstdio>
#include <vector>
#include <cstring>
#include <cmath>
#include "ggml.h"
#include "ggml-backend.h"

static void cpu_roll_ref(const float* src, float* dst,
                         int64_t rows, int64_t cols,
                         int64_t drow, int64_t dcol) {
    auto mod = [](int64_t a, int64_t m){ int64_t r=a%m; return r<0? r+m : r; };
    for (int64_t r=0;r<rows;++r)
      for (int64_t c=0;c<cols;++c){
        int64_t rr = mod(r - drow, rows);
        int64_t cc = mod(c - dcol, cols);
        dst[r*cols + c] = src[rr*cols + cc];
      }
}

static ggml_backend_t init_sycl_backend() {
    ggml_backend_load_all();
    const char* names[] = {"SYCL","SYCL0","sycl","Sycl","ggml-sycl"};
    for (const char* n : names) {
        ggml_backend_t be = ggml_backend_init_by_name(n, NULL);
        if (be) { std::fprintf(stderr, "Initialized backend: %s\n", n); return be; }
    }
    return nullptr;
}

int main() {
    const int64_t rows=4, cols=6;
    const int64_t drow=1, dcol=-2;
    const size_t  N = (size_t)(rows*cols);

    std::vector<float> h_src(N), h_ref(N), h_out(N);
    for (size_t i=0;i<N;++i) h_src[i] = (float)i;
    cpu_roll_ref(h_src.data(), h_ref.data(), rows, cols, drow, dcol);

    // ctx + no-alloc
    ggml_init_params ip = { 8*1024*1024, NULL, false };
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) { std::fprintf(stderr,"ggml_init failed\n"); return 1; }
    ggml_set_no_alloc(ctx, true);

    // שני ROLL חד-ציריים (תואם למימוש ה-minimal)
    ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor* T = ggml_roll(ctx, A, dcol, 0,   0, 0); // עמודות בלבד
    ggml_tensor* Y = ggml_roll(ctx, T, 0,   drow, 0, 0); // שורות בלבד

    // backend + alloc
    ggml_backend_t backend = init_sycl_backend();
    if (!backend) { std::fprintf(stderr,"No SYCL backend\n"); ggml_free(ctx); return 2; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { std::fprintf(stderr,"alloc_ctx_tensors failed\n"); ggml_backend_free(backend); ggml_free(ctx); return 3; }

    // העלאת קלט דרך ה-backend
    ggml_backend_tensor_set(A, h_src.data(), 0, N*sizeof(float));
    ggml_backend_synchronize(backend);

    // גרף כמצביע (תואם ל-API שלך)
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);

    if (!ggml_backend_graph_compute(backend, gf)) {
        std::fprintf(stderr, "graph_compute failed\n");
        ggml_backend_buffer_free(buf); ggml_backend_free(backend); ggml_free(ctx);
        return 5;
    }
    ggml_backend_synchronize(backend);

    ggml_backend_tensor_get(Y, h_out.data(), 0, N*sizeof(float));
    ggml_backend_synchronize(backend);

    int mism=0;
    for (size_t i=0;i<N;++i){
        float a=h_ref[i], b=h_out[i];
        if (std::fabs(a-b)>1e-5f){
            if (mism<16) std::fprintf(stderr,"mismatch i=%zu ref=%g out=%g\n", i,a,b);
            mism++;
        }
    }
    if (!mism) std::printf("SYCL ROLL (two 1D ops) sanity: PASS\n");
    else       std::printf("SYCL ROLL (two 1D ops) sanity: FAIL mism=%d\n", mism);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return mism?7:0;
}
