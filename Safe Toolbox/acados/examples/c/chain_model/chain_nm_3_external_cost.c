/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CODEGEN_PREFIX
  #define NAMESPACE_CONCAT(NS, ID) _NAMESPACE_CONCAT(NS, ID)
  #define _NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) chain_nm_3_external_cost_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_sq CASADI_PREFIX(sq)

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

static const casadi_int casadi_s0[16] = {12, 1, 0, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const casadi_int casadi_s1[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[19] = {15, 1, 0, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
static const casadi_int casadi_s4[33] = {15, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

casadi_real casadi_sq(casadi_real x) { return x*x;}

/* chain_nm_3_external_cost:(i0[12],i1[3])->(o0,o1[15],o2[15x15,15nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=5.0000000000000000e-01;
  a1=arg[1] ? arg[1][0] : 0;
  a2=casadi_sq(a1);
  a2=(a0*a2);
  a3=arg[1] ? arg[1][1] : 0;
  a4=casadi_sq(a3);
  a4=(a0*a4);
  a2=(a2+a4);
  a4=arg[1] ? arg[1][2] : 0;
  a5=casadi_sq(a4);
  a5=(a0*a5);
  a2=(a2+a5);
  a5=5.0000000000000001e-03;
  a6=arg[0] ? arg[0][0] : 0;
  a7=5.0045119999999998e-01;
  a6=(a6-a7);
  a7=casadi_sq(a6);
  a7=(a5*a7);
  a2=(a2+a7);
  a7=arg[0] ? arg[0][1] : 0;
  a8=casadi_sq(a7);
  a8=(a5*a8);
  a2=(a2+a8);
  a8=arg[0] ? arg[0][2] : 0;
  a9=-1.2559460000000000e-01;
  a8=(a8-a9);
  a9=casadi_sq(a8);
  a9=(a5*a9);
  a2=(a2+a9);
  a9=arg[0] ? arg[0][3] : 0;
  a10=casadi_sq(a9);
  a10=(a5*a10);
  a2=(a2+a10);
  a10=arg[0] ? arg[0][4] : 0;
  a11=casadi_sq(a10);
  a11=(a5*a11);
  a2=(a2+a11);
  a11=arg[0] ? arg[0][5] : 0;
  a12=casadi_sq(a11);
  a12=(a5*a12);
  a2=(a2+a12);
  a12=arg[0] ? arg[0][6] : 0;
  a13=9.9977059999999995e-01;
  a12=(a12-a13);
  a13=casadi_sq(a12);
  a13=(a5*a13);
  a2=(a2+a13);
  a13=arg[0] ? arg[0][7] : 0;
  a14=casadi_sq(a13);
  a14=(a5*a14);
  a2=(a2+a14);
  a14=arg[0] ? arg[0][8] : 0;
  a15=6.2792529999999999e-02;
  a14=(a14-a15);
  a15=casadi_sq(a14);
  a15=(a5*a15);
  a2=(a2+a15);
  a15=arg[0] ? arg[0][9] : 0;
  a16=casadi_sq(a15);
  a16=(a5*a16);
  a2=(a2+a16);
  a16=arg[0] ? arg[0][10] : 0;
  a17=casadi_sq(a16);
  a17=(a5*a17);
  a2=(a2+a17);
  a17=arg[0] ? arg[0][11] : 0;
  a18=casadi_sq(a17);
  a18=(a5*a18);
  a2=(a2+a18);
  if (res[0]!=0) res[0][0]=a2;
  a1=(a1+a1);
  a1=(a0*a1);
  if (res[1]!=0) res[1][0]=a1;
  a3=(a3+a3);
  a3=(a0*a3);
  if (res[1]!=0) res[1][1]=a3;
  a4=(a4+a4);
  a0=(a0*a4);
  if (res[1]!=0) res[1][2]=a0;
  a6=(a6+a6);
  a6=(a5*a6);
  if (res[1]!=0) res[1][3]=a6;
  a7=(a7+a7);
  a7=(a5*a7);
  if (res[1]!=0) res[1][4]=a7;
  a8=(a8+a8);
  a8=(a5*a8);
  if (res[1]!=0) res[1][5]=a8;
  a9=(a9+a9);
  a9=(a5*a9);
  if (res[1]!=0) res[1][6]=a9;
  a10=(a10+a10);
  a10=(a5*a10);
  if (res[1]!=0) res[1][7]=a10;
  a11=(a11+a11);
  a11=(a5*a11);
  if (res[1]!=0) res[1][8]=a11;
  a12=(a12+a12);
  a12=(a5*a12);
  if (res[1]!=0) res[1][9]=a12;
  a13=(a13+a13);
  a13=(a5*a13);
  if (res[1]!=0) res[1][10]=a13;
  a14=(a14+a14);
  a14=(a5*a14);
  if (res[1]!=0) res[1][11]=a14;
  a15=(a15+a15);
  a15=(a5*a15);
  if (res[1]!=0) res[1][12]=a15;
  a16=(a16+a16);
  a16=(a5*a16);
  if (res[1]!=0) res[1][13]=a16;
  a17=(a17+a17);
  a5=(a5*a17);
  if (res[1]!=0) res[1][14]=a5;
  a5=1.;
  if (res[2]!=0) res[2][0]=a5;
  if (res[2]!=0) res[2][1]=a5;
  if (res[2]!=0) res[2][2]=a5;
  a5=1.0000000000000000e-02;
  if (res[2]!=0) res[2][3]=a5;
  if (res[2]!=0) res[2][4]=a5;
  if (res[2]!=0) res[2][5]=a5;
  if (res[2]!=0) res[2][6]=a5;
  if (res[2]!=0) res[2][7]=a5;
  if (res[2]!=0) res[2][8]=a5;
  if (res[2]!=0) res[2][9]=a5;
  if (res[2]!=0) res[2][10]=a5;
  if (res[2]!=0) res[2][11]=a5;
  if (res[2]!=0) res[2][12]=a5;
  if (res[2]!=0) res[2][13]=a5;
  if (res[2]!=0) res[2][14]=a5;
  return 0;
}

CASADI_SYMBOL_EXPORT int chain_nm_3_external_cost(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, void* mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT void chain_nm_3_external_cost_incref(void) {
}

CASADI_SYMBOL_EXPORT void chain_nm_3_external_cost_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int chain_nm_3_external_cost_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int chain_nm_3_external_cost_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT const char* chain_nm_3_external_cost_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* chain_nm_3_external_cost_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* chain_nm_3_external_cost_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* chain_nm_3_external_cost_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int chain_nm_3_external_cost_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
