# tir-bench baseline view: `baseline.json`

- Timestamp: `20260608T193110Z-noise-median`
- Label:     `fp8-paged-mqa-logits-full-sweep+failed-retry+noise-median`
- Git:       `{'tir': 'e97cbf48-dirty', 'tirx-kernels': '9d2cf401', 'tirx-bench-ci': '08a57cf3'}`
- Workloads: 160 ok, 0 failed

Each row shows our impl's time (tir/tirx) and every reference impl, with ref/ours where ref = fastest non-ours impl. Higher ratio = ours is faster.

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0404 | deepgemm | 0.0426 | 1.055 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0533 | deepgemm | 0.0557 | 1.045 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0414 | deepgemm | 0.0421 | 1.017 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0535 | deepgemm | 0.0547 | 1.022 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0410 | deepgemm | 0.0438 | 1.067 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0533 | deepgemm | 0.0573 | 1.074 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0401 | deepgemm | 0.0395 | 0.986 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0520 | deepgemm | 0.0515 | 0.989 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.0696 | deepgemm | 0.0733 | 1.054 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1086 | deepgemm | 0.1141 | 1.051 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.0709 | deepgemm | 0.0719 | 1.014 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1082 | deepgemm | 0.1098 | 1.015 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.0714 | deepgemm | 0.0770 | 1.079 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1103 | deepgemm | 0.1206 | 1.093 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 0.0698 | deepgemm | 0.0684 | 0.979 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1064 | deepgemm | 0.1053 | 0.989 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0716 | deepgemm | 0.0761 | 1.064 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0706 | deepgemm | 0.0754 | 1.069 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0713 | deepgemm | 0.0740 | 1.039 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0719 | deepgemm | 0.0748 | 1.040 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0712 | deepgemm | 0.0776 | 1.090 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0719 | deepgemm | 0.0781 | 1.086 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0695 | deepgemm | 0.0696 | 1.002 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0702 | deepgemm | 0.0702 | 1.001 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.1252 | deepgemm | 0.1331 | 1.063 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1806 | deepgemm | 0.1924 | 1.065 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.1262 | deepgemm | 0.1298 | 1.028 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1802 | deepgemm | 0.1844 | 1.023 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.1278 | deepgemm | 0.1403 | 1.098 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1803 | deepgemm | 0.1987 | 1.102 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 0.1251 | deepgemm | 0.1246 | 0.996 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1794 | deepgemm | 0.1788 | 0.997 | — |
## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0451 | deepgemm | 0.0458 | 1.017 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0597 | deepgemm | 0.0605 | 1.014 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0448 | deepgemm | 0.0444 | 0.991 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0580 | deepgemm | 0.0578 | 0.997 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0450 | deepgemm | 0.0467 | 1.039 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0593 | deepgemm | 0.0613 | 1.033 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0437 | deepgemm | 0.0431 | 0.986 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0585 | deepgemm | 0.0577 | 0.987 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.0782 | deepgemm | 0.0799 | 1.021 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1228 | deepgemm | 0.1255 | 1.022 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.0771 | deepgemm | 0.0769 | 0.997 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1193 | deepgemm | 0.1196 | 1.003 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.0778 | deepgemm | 0.0807 | 1.036 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1180 | deepgemm | 0.1223 | 1.037 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 0.0772 | deepgemm | 0.0763 | 0.988 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1170 | deepgemm | 0.1162 | 0.993 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0806 | deepgemm | 0.0823 | 1.022 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0820 | deepgemm | 0.0833 | 1.016 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0769 | deepgemm | 0.0769 | 1.000 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0781 | deepgemm | 0.0777 | 0.996 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0768 | deepgemm | 0.0817 | 1.064 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0786 | deepgemm | 0.0835 | 1.062 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0762 | deepgemm | 0.0760 | 0.997 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0767 | deepgemm | 0.0758 | 0.989 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.1415 | deepgemm | 0.1451 | 1.026 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.2013 | deepgemm | 0.2038 | 1.013 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.1359 | deepgemm | 0.1367 | 1.006 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1960 | deepgemm | 0.1966 | 1.003 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.1339 | deepgemm | 0.1445 | 1.079 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1955 | deepgemm | 0.2072 | 1.060 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 0.1334 | deepgemm | 0.1333 | 0.999 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1919 | deepgemm | 0.1920 | 1.000 | — |
## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0045 | deepgemm | 0.0045 | 0.987 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0045 | deepgemm | 0.0044 | 0.992 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.992 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.997 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0041 | deepgemm | 0.0040 | 0.980 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.987 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0041 | deepgemm | 0.0040 | 0.986 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.991 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.992 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0041 | deepgemm | 0.0040 | 0.988 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.996 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 1.002 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.995 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.995 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 1.002 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.000 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.996 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.999 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.001 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.003 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 1.000 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.994 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.988 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0041 | deepgemm | 0.0041 | 0.995 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.992 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0041 | deepgemm | 0.0040 | 0.991 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.994 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.995 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0041 | deepgemm | 0.0040 | 0.988 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.993 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0041 | deepgemm | 0.0040 | 0.984 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.998 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.993 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0045 | deepgemm | 0.0045 | 0.994 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.991 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.995 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.998 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.998 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.994 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.994 | — |
## flash_attention4

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 0.0209 | flashattn_sm100 | 0.0204 | 0.976 | flashinfer=0.0261 |
| `s1024_h32kv16_causal` | tir | 0.0205 | flashattn_sm100 | 0.0198 | 0.965 | flashinfer=0.0260 |
| `s1024_h32kv32` | tir | 0.0212 | flashattn_sm100 | 0.0206 | 0.971 | flashinfer=0.0261 |
| `s1024_h32kv32_causal` | tir | 0.0223 | flashattn_sm100 | 0.0205 | 0.922 | flashinfer=0.0262 |
| `s1024_h32kv4` | tir | 0.0208 | flashattn_sm100 | 0.0202 | 0.971 | flashinfer=0.0260 |
| `s1024_h32kv4_causal` | tir | 0.0197 | flashattn_sm100 | 0.0196 | 0.996 | flashinfer=0.0259 |
| `s1024_h32kv8` | tir | 0.0210 | flashattn_sm100 | 0.0204 | 0.972 | flashinfer=0.0262 |
| `s1024_h32kv8_causal` | tir | 0.0201 | flashattn_sm100 | 0.0197 | 0.983 | flashinfer=0.0260 |
| `s2048_h32kv16` | tir | 0.0613 | flashattn_sm100 | 0.0594 | 0.970 | flashinfer=0.0775 |
| `s2048_h32kv16_causal` | tir | 0.0383 | flashattn_sm100 | 0.0392 | 1.023 | flashinfer=0.0775 |
| `s2048_h32kv32` | tir | 0.0629 | flashattn_sm100 | 0.0610 | 0.970 | flashinfer=0.0778 |
| `s2048_h32kv32_causal` | tir | 0.0422 | flashattn_sm100 | 0.0400 | 0.948 | flashinfer=0.0774 |
| `s2048_h32kv4` | tir | 0.0608 | flashattn_sm100 | 0.0588 | 0.968 | flashinfer=0.0776 |
| `s2048_h32kv4_causal` | tir | 0.0379 | flashattn_sm100 | 0.0390 | 1.030 | flashinfer=0.0783 |
| `s2048_h32kv8` | tir | 0.0620 | flashattn_sm100 | 0.0598 | 0.966 | flashinfer=0.0787 |
| `s2048_h32kv8_causal` | tir | 0.0379 | flashattn_sm100 | 0.0387 | 1.022 | flashinfer=0.0774 |
| `s4096_h32kv16` | tir | 0.2153 | flashattn_sm100 | 0.2124 | 0.987 | flashinfer=0.2688 |
| `s4096_h32kv16_causal` | tir | 0.1147 | flashattn_sm100 | 0.1165 | 1.015 | flashinfer=0.2686 |
| `s4096_h32kv32` | tir | 0.2160 | flashattn_sm100 | 0.2120 | 0.981 | flashinfer=0.2698 |
| `s4096_h32kv32_causal` | tir | 0.1232 | flashattn_sm100 | 0.1173 | 0.952 | flashinfer=0.2673 |
| `s4096_h32kv4` | tir | 0.2061 | flashattn_sm100 | 0.2014 | 0.977 | flashinfer=0.2654 |
| `s4096_h32kv4_causal` | tir | 0.1138 | flashattn_sm100 | 0.1142 | 1.004 | flashinfer=0.2689 |
| `s4096_h32kv8` | tir | 0.2110 | flashattn_sm100 | 0.2068 | 0.980 | flashinfer=0.2682 |
| `s4096_h32kv8_causal` | tir | 0.1150 | flashattn_sm100 | 0.1160 | 1.009 | flashinfer=0.2706 |
| `s8192_h32kv16` | tir | 0.7486 | flashattn_sm100 | 0.7349 | 0.982 | flashinfer=0.9012 |
| `s8192_h32kv16_causal` | tir | 0.4051 | flashattn_sm100 | 0.4020 | 0.992 | flashinfer=0.8958 |
| `s8192_h32kv32` | tir | 0.7528 | flashattn_sm100 | 0.7443 | 0.989 | flashinfer=0.9062 |
| `s8192_h32kv32_causal` | tir | 0.4311 | flashattn_sm100 | 0.4133 | 0.959 | flashinfer=0.9138 |
| `s8192_h32kv4` | tir | 0.7516 | flashattn_sm100 | 0.7293 | 0.970 | flashinfer=0.8930 |
| `s8192_h32kv4_causal` | tir | 0.3981 | flashattn_sm100 | 0.3950 | 0.992 | flashinfer=0.8906 |
| `s8192_h32kv8` | tir | 0.7560 | flashattn_sm100 | 0.7388 | 0.977 | flashinfer=0.9151 |
| `s8192_h32kv8_causal` | tir | 0.4023 | flashattn_sm100 | 0.4018 | 0.999 | flashinfer=0.9021 |
## fp16_bf16_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 0.0165 | torch-cublas | 0.0055 | 0.332 | deepgemm-bf16=0.0078, deepgemm-cublaslt=0.0055 |
| `bf16_16384x16384x16384` | tir | 5.9831 | deepgemm-cublaslt | 5.7802 | 0.966 | deepgemm-bf16=7.3368, torch-cublas=5.8139 |
| `bf16_2048x2048x2048` | tir | 0.0273 | torch-cublas | 0.0182 | 0.668 | deepgemm-bf16=0.0188, deepgemm-cublaslt=0.0183 |
| `bf16_4096x4096x4096` | tir | 0.0908 | deepgemm-bf16 | 0.0899 | 0.990 | deepgemm-cublaslt=0.0915, torch-cublas=0.0917 |
| `bf16_8192x8192x8192` | tir | 0.6705 | deepgemm-cublaslt | 0.6975 | 1.040 | deepgemm-bf16=0.7066, torch-cublas=0.6977 |
| `fp16_1024x1024x1024` | tir | 0.0165 | deepgemm-cublaslt | 0.0055 | 0.332 | torch-cublas=0.0055 |
| `fp16_16384x16384x16384` | tir | 6.3561 | torch-cublas | 5.9995 | 0.944 | deepgemm-cublaslt=6.0711 |
| `fp16_2048x2048x2048` | tir | 0.0277 | deepgemm-cublaslt | 0.0187 | 0.674 | torch-cublas=0.0187 |
| `fp16_4096x4096x4096` | tir | 0.0941 | torch-cublas | 0.0956 | 1.016 | deepgemm-cublaslt=0.0956 |
| `fp16_8192x8192x8192` | tir | 0.6865 | torch-cublas | 0.7105 | 1.035 | deepgemm-cublaslt=0.7164 |
## fp8_blockwise_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 0.0489 | deepgemm | 0.0500 | 1.023 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 0.1148 | deepgemm | 0.1160 | 1.010 | — |
| `deepgemm_m4096_n32768_k512` | tir | 0.0720 | deepgemm | 0.0757 | 1.052 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 0.0835 | deepgemm | 0.0842 | 1.008 | — |
| `deepgemm_m4096_n576_k7168` | tir | 0.0186 | deepgemm | 0.0194 | 1.045 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 0.3233 | deepgemm | 0.3232 | 1.000 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 0.0436 | deepgemm | 0.0446 | 1.023 | — |
| `smoke_1024x1024x1024` | tir | 0.0059 | deepgemm | 0.0068 | 1.161 | — |
| `stress_m8192_n7168_k4096` | tir | 0.1624 | deepgemm | 0.1629 | 1.003 | — |
## nvfp4_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 0.0067 | flashinfer | 0.0052 | 0.782 | — |
| `16384x16384x16384` | tir | 1.4373 | flashinfer | 1.5839 | 1.102 | — |
| `2048x2048x2048` | tir | 0.0088 | flashinfer | 0.0084 | 0.956 | — |
| `4096x4096x4096` | tir | 0.0303 | flashinfer | 0.0319 | 1.054 | — |
| `8192x8192x8192` | tir | 0.1870 | flashinfer | 0.1758 | 0.940 | — |
