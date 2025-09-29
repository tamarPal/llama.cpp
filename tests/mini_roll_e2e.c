#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

#include "ggml.h"
#include "ggml-backend.h"

/* עוזר קטן לטעינה דינמית */
static void must_dlopen(const char *path) {
    if (!path || !*path) return;
    void *h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        fprintf(stderr, "dlopen failed for %s: %s\n", path, dlerror());
        exit(4);
    }
}

static void fill_iota_f32(float *p, int n) { for (int i=0;i<n;++i) p[i]=(float)i; }

static int almost_equal(const float *a,const float *b,int n,float atol,float rtol){
    for (int i=0;i<n;++i){
        float d=fabsf(a[i]-b[i]), t=atol+rtol*fmaxf(fabsf(a[i]),fabsf(b[i]));
        if (d>t){ fprintf(stderr,"mismatch @%d a=%g b=%g\n",i,a[i],b[i]); return 0; }
    }
    return 1;
}

/* אצלך: ggml_roll(ctx, a, shift0..3) */
static struct ggml_tensor * make_roll_graph(struct ggml_context *ctx, struct ggml_tensor *x,
                                            int axis, int shift) {
    int s0=0,s1=0,s2=0,s3=0;
    if (axis==0) s0=shift; else if (axis==1) s1=shift; else if (axis==2) s2=shift; else if (axis==3) s3=shift;
    return ggml_roll(ctx, x, s0, s1, s2, s3);
}

static ggml_backend_t init_backend_by_name(const char *name) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name(name);
    if (!reg) { fprintf(stderr,"backend registry '%s' not found\n", name); return NULL; }
    size_t ndev = ggml_backend_reg_dev_count(reg);
    if (ndev == 0) { fprintf(stderr,"no devices for '%s'\n", name); return NULL; }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    if (!dev) { fprintf(stderr,"reg_dev_get failed for '%s'\n", name); return NULL; }
    ggml_backend_t be = ggml_backend_dev_init(dev, NULL);
    if (!be) { fprintf(stderr,"dev_init failed for '%s'\n", name); return NULL; }
    return be;
}

static void run_once(const char *backend_name, int ne0,int ne1,int axis,int shift, float *out_buf){
    struct ggml_init_params ip = { 64*1024*1024, NULL, false };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    fill_iota_f32((float*)x->data, ne0*ne1);
    struct ggml_tensor *y = make_roll_graph(ctx, x, axis, shift);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t backend = init_backend_by_name(backend_name);
    if (!backend) exit(3);
    if (!ggml_backend_graph_compute(backend, gf)) {
        fprintf(stderr,"graph compute failed on %s\n", backend_name);
        exit(2);
    }
    memcpy(out_buf, (const float*)y->data, sizeof(float)*ne0*ne1);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

int main(void) {
    /* טוענים ספריות לפי ENV אם ניתנו */
    const char *cpu_so  = getenv("GGML_CPU_SO");
    const char *sycl_so = getenv("GGML_SYCL_SO");
    if (cpu_so)  must_dlopen(cpu_so);
    if (sycl_so) must_dlopen(sycl_so);

    const int cases[][4]={{8,1,0,1},{8,1,0,-2},{7,3,0,3},{7,3,1,2},{5,5,1,7},{9,4,0,16}};
    for (int i=0;i<(int)(sizeof(cases)/sizeof(cases[0]));++i){
        int ne0=cases[i][0], ne1=cases[i][1], axis=cases[i][2], shift=cases[i][3], N=ne0*ne1;
        float *cpu_out=(float*)malloc(sizeof(float)*N), *sycl_out=(float*)malloc(sizeof(float)*N);

        run_once("CPU",  ne0,ne1,axis,shift, cpu_out);
        run_once("SYCL", ne0,ne1,axis,shift, sycl_out);

        int ok = almost_equal(cpu_out, sycl_out, N, 1e-6f, 0.0f);
        printf("CASE %d: ne=(%d,%d) axis=%d shift=%d -> %s\n", i, ne0, ne1, axis, shift, ok?"OK":"MISMATCH");
        if (!ok) { for (int j=0;j<N && j<32;++j) printf("  i=%d cpu=%g sycl=%g\n", j, cpu_out[j], sycl_out[j]); return 1; }

        free(cpu_out); free(sycl_out);
    }
    puts("ALL CASES PASSED");
    return 0;
}
