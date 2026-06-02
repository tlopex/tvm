# tir-bench baseline view: `baseline.json`

- Timestamp: `20260602T022612Z`
- Label:     `kernel-cleanup-full`
- Git:       `{'tir': '1dbe1d44-dirty', 'tirx-kernels': '90a1ecb2', 'tirx-bench-ci': '08a57cf3'}`
- Workloads: 118 ok, 2 failed

Each row shows our impl's time (tir/tirx) and every reference impl, with ref/ours where ref = fastest non-ours impl. Higher ratio = ours is faster.

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0410 | deepgemm | 0.0428 | 1.043 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0525 | deepgemm | 0.0552 | 1.052 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0415 | deepgemm | 0.0421 | 1.014 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0531 | deepgemm | 0.0542 | 1.021 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0415 | deepgemm | 0.0444 | 1.069 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0531 | deepgemm | 0.0571 | 1.075 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0402 | deepgemm | 0.0396 | 0.987 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0522 | deepgemm | 0.0516 | 0.989 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.0698 | deepgemm | 0.0733 | 1.049 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1069 | deepgemm | 0.1132 | 1.059 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.0712 | deepgemm | 0.0721 | 1.013 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1093 | deepgemm | 0.1109 | 1.014 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.0713 | deepgemm | 0.0771 | 1.081 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1086 | deepgemm | 0.1186 | 1.092 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 0.0700 | deepgemm | 0.0688 | 0.982 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1057 | deepgemm | 0.1044 | 0.988 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0715 | deepgemm | 0.0765 | 1.069 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0706 | deepgemm | 0.0754 | 1.068 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0717 | deepgemm | 0.0745 | 1.039 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0721 | deepgemm | 0.0749 | 1.039 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0708 | deepgemm | 0.0772 | 1.089 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0715 | deepgemm | 0.0778 | 1.089 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0704 | deepgemm | 0.0705 | 1.002 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0703 | deepgemm | 0.0704 | 1.001 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.1243 | deepgemm | 0.1325 | 1.066 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1773 | deepgemm | 0.1891 | 1.067 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.1256 | deepgemm | 0.1288 | 1.026 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1797 | deepgemm | 0.1841 | 1.024 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.1258 | deepgemm | 0.1382 | 1.099 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1799 | deepgemm | 0.1986 | 1.104 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 0.1227 | deepgemm | 0.1224 | 0.998 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1768 | deepgemm | 0.1761 | 0.996 | — |
## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0458 | deepgemm | 0.0465 | 1.014 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0600 | deepgemm | 0.0606 | 1.009 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0444 | deepgemm | 0.0441 | 0.994 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0580 | deepgemm | 0.0578 | 0.997 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0445 | deepgemm | 0.0465 | 1.045 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0587 | deepgemm | 0.0602 | 1.026 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0443 | deepgemm | 0.0434 | 0.980 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0577 | deepgemm | 0.0566 | 0.981 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.0787 | deepgemm | 0.0805 | 1.022 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1219 | deepgemm | 0.1241 | 1.018 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.0781 | deepgemm | 0.0773 | 0.990 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1192 | deepgemm | 0.1201 | 1.008 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.0777 | deepgemm | 0.0808 | 1.039 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1180 | deepgemm | 0.1223 | 1.036 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 0.0770 | deepgemm | 0.0768 | 0.998 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1184 | deepgemm | 0.1178 | 0.994 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0809 | deepgemm | 0.0825 | 1.019 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0800 | deepgemm | 0.0817 | 1.021 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0770 | deepgemm | 0.0769 | 0.998 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0767 | deepgemm | 0.0766 | 0.998 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0766 | deepgemm | 0.0815 | 1.064 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0763 | deepgemm | 0.0813 | 1.066 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0763 | deepgemm | 0.0758 | 0.993 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0767 | deepgemm | 0.0768 | 1.002 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.1411 | deepgemm | 0.1447 | 1.025 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1999 | deepgemm | 0.2041 | 1.021 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.1372 | deepgemm | 0.1380 | 1.005 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1968 | deepgemm | 0.1971 | 1.001 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.1362 | deepgemm | 0.1448 | 1.063 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1927 | deepgemm | 0.2041 | 1.060 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 0.1327 | deepgemm | 0.1334 | 1.006 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1907 | deepgemm | 0.1912 | 1.003 | — |
## flash_attention4

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 0.0210 | flashattn_sm100 | 0.0203 | 0.966 | flashinfer=0.0260 |
| `s1024_h32kv16_causal` | tir | 0.0205 | flashattn_sm100 | 0.0199 | 0.968 | flashinfer=0.0260 |
| `s1024_h32kv32` | tir | 0.0212 | flashattn_sm100 | 0.0205 | 0.967 | flashinfer=0.0261 |
| `s1024_h32kv32_causal` | tir | 0.0224 | flashattn_sm100 | 0.0206 | 0.918 | flashinfer=0.0263 |
| `s1024_h32kv4` | tir | 0.0208 | flashattn_sm100 | 0.0202 | 0.972 | flashinfer=0.0260 |
| `s1024_h32kv4_causal` | tir | 0.0194 | flashattn_sm100 | 0.0194 | 0.999 | flashinfer=0.0259 |
| `s1024_h32kv8` | tir | 0.0208 | flashattn_sm100 | 0.0204 | 0.977 | flashinfer=0.0259 |
| `s1024_h32kv8_causal` | tir | 0.0196 | flashattn_sm100 | 0.0196 | 0.997 | flashinfer=0.0259 |
| `s2048_h32kv16` | tir | 0.0622 | flashattn_sm100 | 0.0601 | 0.966 | flashinfer=0.0772 |
| `s2048_h32kv16_causal` | tir | 0.0387 | flashattn_sm100 | 0.0394 | 1.019 | flashinfer=0.0776 |
| `s2048_h32kv32` | tir | 0.0627 | flashattn_sm100 | 0.0609 | 0.971 | flashinfer=0.0778 |
| `s2048_h32kv32_causal` | tir | 0.0419 | flashattn_sm100 | 0.0399 | 0.953 | flashinfer=0.0771 |
| `s2048_h32kv4` | tir | 0.0603 | flashattn_sm100 | 0.0584 | 0.969 | flashinfer=0.0774 |
| `s2048_h32kv8` | tir | 0.0611 | flashattn_sm100 | 0.0590 | 0.966 | flashinfer=0.0775 |
| `s2048_h32kv8_causal` | tir | 0.0376 | flashattn_sm100 | 0.0383 | 1.019 | flashinfer=0.0767 |
| `s4096_h32kv16` | tir | 0.2138 | flashattn_sm100 | 0.2088 | 0.977 | flashinfer=0.2658 |
| `s4096_h32kv16_causal` | tir | 0.1138 | flashattn_sm100 | 0.1155 | 1.015 | flashinfer=0.2661 |
| `s4096_h32kv32` | tir | 0.2181 | flashattn_sm100 | 0.2171 | 0.995 | flashinfer=0.2738 |
| `s4096_h32kv32_causal` | tir | 0.1255 | flashattn_sm100 | 0.1199 | 0.955 | flashinfer=0.2719 |
| `s4096_h32kv4` | tir | 0.2070 | flashattn_sm100 | 0.2026 | 0.979 | flashinfer=0.2663 |
| `s4096_h32kv4_causal` | tir | 0.1128 | flashattn_sm100 | 0.1135 | 1.006 | flashinfer=0.2664 |
| `s4096_h32kv8_causal` | tir | 0.1127 | flashattn_sm100 | 0.1142 | 1.013 | flashinfer=0.2656 |
| `s8192_h32kv16` | tir | 0.7388 | flashattn_sm100 | 0.7284 | 0.986 | flashinfer=0.8972 |
| `s8192_h32kv16_causal` | tir | 0.4090 | flashattn_sm100 | 0.4079 | 0.997 | flashinfer=0.9008 |
| `s8192_h32kv32` | tir | 0.7509 | flashattn_sm100 | 0.7439 | 0.991 | flashinfer=0.9082 |
| `s8192_h32kv32_causal` | tir | 0.4272 | flashattn_sm100 | 0.4104 | 0.961 | flashinfer=0.9039 |
| `s8192_h32kv4` | tir | 0.7637 | flashattn_sm100 | 0.7379 | 0.966 | flashinfer=0.9146 |
| `s8192_h32kv4_causal` | tir | 0.4002 | flashattn_sm100 | 0.4016 | 1.003 | flashinfer=0.8934 |
| `s8192_h32kv8` | tir | 0.7433 | flashattn_sm100 | 0.7301 | 0.982 | flashinfer=0.8979 |
| `s8192_h32kv8_causal` | tir | 0.4017 | flashattn_sm100 | 0.4032 | 1.004 | flashinfer=0.8967 |
## fp16_bf16_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 0.0165 | deepgemm-cublaslt | 0.0055 | 0.333 | deepgemm-bf16=0.0075, torch-cublas=0.0055 |
| `bf16_16384x16384x16384` | tir | 5.8562 | torch-cublas | 5.6006 | 0.956 | deepgemm-bf16=6.2162, deepgemm-cublaslt=5.6894 |
| `bf16_2048x2048x2048` | tir | 0.0274 | torch-cublas | 0.0185 | 0.677 | deepgemm-bf16=0.0187, deepgemm-cublaslt=0.0186 |
| `bf16_4096x4096x4096` | tir | 0.0911 | deepgemm-bf16 | 0.0914 | 1.003 | deepgemm-cublaslt=0.0924, torch-cublas=0.0927 |
| `bf16_8192x8192x8192` | tir | 0.6621 | deepgemm-cublaslt | 0.6899 | 1.042 | deepgemm-bf16=0.6969, torch-cublas=0.6915 |
| `fp16_1024x1024x1024` | tir | 0.0165 | torch-cublas | 0.0055 | 0.333 | deepgemm-cublaslt=0.0056 |
| `fp16_16384x16384x16384` | tir | 6.4118 | deepgemm-cublaslt | 5.9586 | 0.929 | torch-cublas=6.0401 |
| `fp16_2048x2048x2048` | tir | 0.0273 | torch-cublas | 0.0185 | 0.678 | deepgemm-cublaslt=0.0185 |
| `fp16_4096x4096x4096` | tir | 0.0938 | torch-cublas | 0.0949 | 1.013 | deepgemm-cublaslt=0.0950 |
| `fp16_8192x8192x8192` | tir | 0.6803 | torch-cublas | 0.7094 | 1.043 | deepgemm-cublaslt=0.7100 |
## fp8_blockwise_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 0.0485 | deepgemm | 0.0495 | 1.021 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 0.1167 | deepgemm | 0.1180 | 1.011 | — |
| `deepgemm_m4096_n32768_k512` | tir | 0.0727 | deepgemm | 0.0768 | 1.057 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 0.0830 | deepgemm | 0.0838 | 1.009 | — |
| `deepgemm_m4096_n576_k7168` | tir | 0.0186 | deepgemm | 0.0194 | 1.042 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 0.3219 | deepgemm | 0.3199 | 0.994 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 0.0434 | deepgemm | 0.0443 | 1.022 | — |
| `smoke_1024x1024x1024` | tir | 0.0059 | deepgemm | 0.0068 | 1.158 | — |
| `stress_m8192_n7168_k4096` | tir | 0.1634 | deepgemm | 0.1648 | 1.008 | — |
## nvfp4_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 0.0066 | flashinfer | 0.0051 | 0.774 | — |
| `16384x16384x16384` | tir | 1.4392 | flashinfer | 1.5592 | 1.083 | — |
| `2048x2048x2048` | tir | 0.0088 | flashinfer | 0.0083 | 0.947 | — |
| `4096x4096x4096` | tir | 0.0301 | flashinfer | 0.0317 | 1.051 | — |
| `8192x8192x8192` | tir | 0.1869 | flashinfer | 0.1752 | 0.938 | — |

## Failed (2)

- `flash_attention4/s2048_h32kv4_causal`: INTERFERED on all 5 attempts (last intruders: [3505275])
- `flash_attention4/s4096_h32kv8`: INTERFERED on all 5 attempts (last intruders: [3505234])
