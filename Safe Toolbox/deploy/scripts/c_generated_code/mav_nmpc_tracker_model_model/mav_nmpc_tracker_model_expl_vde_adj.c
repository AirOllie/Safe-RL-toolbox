/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) mav_nmpc_tracker_model_expl_vde_adj_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

static const casadi_int casadi_s0[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[7] = {5, 1, 0, 3, 2, 3, 4};

/* mav_nmpc_tracker_model_expl_vde_adj:(i0[3],i1[3],i2[2],i3[])->(o0[5x1,3nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, w1, w2, w3, *w4=w+4, w5, w6, w7, w8;
  /* #0: @0 = input[0][2] */
  w0 = arg[0] ? arg[0][2] : 0;
  /* #1: @1 = cos(@0) */
  w1 = cos( w0 );
  /* #2: @2 = 0.5 */
  w2 = 5.0000000000000000e-01;
  /* #3: @3 = input[2][0] */
  w3 = arg[2] ? arg[2][0] : 0;
  /* #4: @2 = (@2+@3) */
  w2 += w3;
  /* #5: @4 = input[1][0] */
  casadi_copy(arg[1], 3, w4);
  /* #6: {@5, @6, @7} = vertsplit(@4) */
  w5 = w4[0];
  w6 = w4[1];
  w7 = w4[2];
  /* #7: @2 = (@2*@6) */
  w2 *= w6;
  /* #8: @1 = (@1*@2) */
  w1 *= w2;
  /* #9: @2 = sin(@0) */
  w2 = sin( w0 );
  /* #10: @8 = 0.5 */
  w8 = 5.0000000000000000e-01;
  /* #11: @8 = (@8+@3) */
  w8 += w3;
  /* #12: @8 = (@8*@5) */
  w8 *= w5;
  /* #13: @2 = (@2*@8) */
  w2 *= w8;
  /* #14: @1 = (@1-@2) */
  w1 -= w2;
  /* #15: output[0][0] = @1 */
  if (res[0]) res[0][0] = w1;
  /* #16: @1 = sin(@0) */
  w1 = sin( w0 );
  /* #17: @1 = (@1*@6) */
  w1 *= w6;
  /* #18: @0 = cos(@0) */
  w0 = cos( w0 );
  /* #19: @0 = (@0*@5) */
  w0 *= w5;
  /* #20: @1 = (@1+@0) */
  w1 += w0;
  /* #21: output[0][1] = @1 */
  if (res[0]) res[0][1] = w1;
  /* #22: output[0][2] = @7 */
  if (res[0]) res[0][2] = w7;
  return 0;
}

CASADI_SYMBOL_EXPORT int mav_nmpc_tracker_model_expl_vde_adj(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int mav_nmpc_tracker_model_expl_vde_adj_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int mav_nmpc_tracker_model_expl_vde_adj_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void mav_nmpc_tracker_model_expl_vde_adj_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int mav_nmpc_tracker_model_expl_vde_adj_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void mav_nmpc_tracker_model_expl_vde_adj_release(int mem) {
}

CASADI_SYMBOL_EXPORT void mav_nmpc_tracker_model_expl_vde_adj_incref(void) {
}

CASADI_SYMBOL_EXPORT void mav_nmpc_tracker_model_expl_vde_adj_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int mav_nmpc_tracker_model_expl_vde_adj_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int mav_nmpc_tracker_model_expl_vde_adj_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real mav_nmpc_tracker_model_expl_vde_adj_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* mav_nmpc_tracker_model_expl_vde_adj_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* mav_nmpc_tracker_model_expl_vde_adj_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* mav_nmpc_tracker_model_expl_vde_adj_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* mav_nmpc_tracker_model_expl_vde_adj_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int mav_nmpc_tracker_model_expl_vde_adj_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 11;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif