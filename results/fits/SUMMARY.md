# Functional-form fits

Fitted with `chaos_llm.curvefit`. BIC comparison; a model within dBIC<2 of the
best is listed as a tie. `domain` says where the fit was made (log domain for
pure exponential/power when the curve spans decades).

| curve | model | best form | parameters | R2 | dBIC(2nd) | note |
|---|---|---|---|---|---|---|
| depth_lam1_median | DeepSeek-1.5B | `c + a*(x-x0)^b` | c=0.0219±0.0074, a=3.03±0.13, b=-1.05±0.029, x0=-1.07±0.044 | 0.9998 | 2.0 | median over 8 tokens of the finite-time depth exponent |
| depth_lam1_t0 | DeepSeek-1.5B | `c + a*(x-x0)^b` | c=0.0357±0.0065, a=3.68±0.14, b=-1.1±0.026, x0=-1.16±0.039 | 0.9999 | 4.9 | token t=0 |
| depth_lam1_t16 | DeepSeek-1.5B | `c + a*exp(-(x/tau)^beta)` | c=0.169±0.0048, a=3.07±0.013, tau=1.62±0.02, beta=0.614±0.0078 | 0.9997 | 11.4 | token t=16 |
| depth_lam1_t50 | DeepSeek-1.5B | `c + a*x^b/(k^b + x^b)` | c=2.28±0.0056, a=-2.29±0.0075, b=0.963±0.011, k=1.02±0.0092 | 0.9999 | 3.2 | token t=50 |
| depth_lam1_t100 | DeepSeek-1.5B | `c + a*exp(-(x/tau)^beta)` | c=0.1±0.0023, a=2.3±0.0064, tau=1.61±0.013, beta=0.625±0.0052 | 0.9999 | 23.3 | token t=100 |
| depth_lam1_t123 | DeepSeek-1.5B | `c + a*(x-x0)^b` | c=0.0153±0.005, a=3.08±0.095, b=-1.07±0.021, x0=-1.1±0.031 | 0.9999 | 4.9 | token t=123 |
| depth_lam1_t143 | DeepSeek-1.5B | `c + a*x^b/(k^b + x^b)` | c=2.78±0.0067, a=-2.77±0.0087, b=0.991±0.011, k=1±0.0088 | 0.9999 | 28.1 | token t=143 |
| depth_lam1_t200 | DeepSeek-1.5B | `c + a*(x-x0)^b` | c=0.0281±0.0079, a=3.27±0.16, b=-1.08±0.032, x0=-1.13±0.049 | 0.9998 | 3.4 | token t=200 |
| depth_lam1_t250 | DeepSeek-1.5B | `c + a*(x-x0)^b` | c=0.0286±0.0082, a=3.38±0.19, b=-1.11±0.037, x0=-1.17±0.056 | 0.9998 | 3.7 | token t=250 |
| depth_lam2_median | DeepSeek-1.5B | `c + a*(x-x0)^b` | c=0.025±0.0058, a=3.11±0.1, b=-1.06±0.023, x0=-1.09±0.035 | 0.9999 | 3.6 | exponent 2, median over tokens |
| depth_lam3_median | DeepSeek-1.5B | `c + a*(x-x0)^b` | c=0.0137±0.007, a=3.02±0.11, b=-1.03±0.027, x0=-1.07±0.04 | 0.9999 | 0.8 | exponent 3, median over tokens |
| depthgain_DeepSeek-1.5B_t5 | DeepSeek-1.5B | `c + a1*exp(-x/t1) + a2*exp(x/t2)` | c=0.413±0.0089, a1=-0.684±0.1, t1=1.18±0.18, a2=0.00489±0.00094, t2=5.09±0.18 | 0.9966 | 27.1 | depth gain, t=5 |
| relresp_DeepSeek-1.5B_t5 | DeepSeek-1.5B | `c + a*(1-exp(-x/t1))*exp(-x/t2)` | c=-0.0194±0.028, a=0.0265±0.028, t1=0.457±0.23, t2=150±1.8e+02 | 0.9494 | 4.3 | relative response, t=5 |
| hprofile_DeepSeek-1.5B_t5 | DeepSeek-1.5B | `a + b*x + c*x^2 + d*x^3` | a=0.5±7, b=24.3±2.3, c=-2.34±0.2, d=0.078±0.0048 | 0.9933 | 3.5 | state norm, t=5 |
| depthgain_DeepSeek-1.5B_t50 | DeepSeek-1.5B | `c + a1*exp(-x/t1) + a2*exp(x/t2)` | c=248±0.0046, a1=-248±0.0051, t1=5.47e+03±5.9e-05, a2=8.57e-07±7.6e-07, t2=1.9±0.12 | 0.9954 | 48.9 | depth gain, t=50 |
| relresp_DeepSeek-1.5B_t50 | DeepSeek-1.5B | `a + b*x + c*x^2` | a=0.00171±0.00039, b=0.000624±6.4e-05, c=-2.12e-05±2.2e-06 | 0.8002 | 1.5 | relative response, t=50 |
| hprofile_DeepSeek-1.5B_t50 | DeepSeek-1.5B | `c + a1*exp(-x/t1) + a2*exp(x/t2)` | c=68.5±5.5, a1=-65.7±6.7, t1=3.51±0.91, a2=1.25±0.25, t2=4.47±0.15 | 0.9981 | 43.0 | state norm, t=50 |
| depthgain_Qwen3-4B_t5 | Qwen3-4B | `a + b*x + c*x^2 + d*x^3` | a=-0.0963±0.042, b=0.114±0.0099, c=-0.00846±0.00063, d=0.000257±1.2e-05 | 0.9981 | 27.4 | depth gain, t=5 |
| relresp_Qwen3-4B_t5 | Qwen3-4B | `a + b*x + c*x^2` | a=0.00779±0.00031, b=0.000435±4e-05, c=-1.23e-05±1.1e-06 | 0.8022 | 1.1 | relative response, t=5 |
| hprofile_Qwen3-4B_t5 | Qwen3-4B | `a*exp(b*x)` | a=9.63±1.4, b=0.112±0.0073 | 0.9708 | 23.8 | state norm, t=5 |
| depthgain_Qwen3-4B_t50 | Qwen3-4B | `c + a1*exp(-x/t1) + a2*exp(x/t2)` | c=0.35±0.057, a1=-0.521±0.11, t1=3.57±1.5, a2=0.0324±0.0057, t2=6.93±0.24 | 0.9975 | 10.0 | depth gain, t=50 |
| relresp_Qwen3-4B_t50 | Qwen3-4B | `a + b*x + c*x^2` | a=0.00469±0.00058, b=0.0011±7.5e-05, c=-2.81e-05±2e-06 | 0.8722 | 2.4 | relative response, t=50 |
| hprofile_Qwen3-4B_t50 | Qwen3-4B | `c + a1*exp(-x/t1) + a2*exp(x/t2)` | c=41.8±6.7, a1=-40±7.6, t1=5.92±3.1, a2=0.265±0.076, t2=4.58±0.17 | 0.9962 | 77.1 | state norm, t=50 |
| depthgain_Phi-4-mini_t5 | Phi-4-mini | `a + b*x + c*x^2 + d*x^3` | a=-0.0452±0.031, b=0.0425±0.0083, c=-0.00405±0.0006, d=0.000153±1.2e-05 | 0.9954 | 84.2 | depth gain, t=5 |
| relresp_Phi-4-mini_t5 | Phi-4-mini | `c + a1*exp(-x/t1) + a2*exp(x/t2)` | c=0.00379±0.00034, a1=-0.00444±0.0013, t1=1.87±0.77, a2=9.47e-05±7.7e-05, t2=7.65±1.5 | 0.9440 | 0.5 | relative response, t=5 |
| hprofile_Phi-4-mini_t5 | Phi-4-mini | `a*exp(b*x)` | a=10.6±0.52, b=0.101±0.0027 | 0.9854 | 229.1 | state norm, t=5 |
| depthgain_Phi-4-mini_t50 | Phi-4-mini | `c + a1*exp(-x/t1) + a2*exp(x/t2)` | c=1.31e+03±7e+02, a1=-1.31e+03±7e+02, t1=3.78e+04±49, a2=0.0497±0.02, t2=5.65±0.38 | 0.9965 | 23.4 | depth gain, t=50 |
| relresp_Phi-4-mini_t50 | Phi-4-mini | `a + b*x` | a=-0.00184±0.00076, b=0.00126±4.1e-05 | 0.9697 | 0.5 | relative response, t=50 |
| hprofile_Phi-4-mini_t50 | Phi-4-mini | `a*exp(b*x)` | a=11.1±0.63, b=0.11±0.0031 | 0.9917 | 232.8 | state norm, t=50 |
| depthgain_Qwen3.5-2B_t5 | Qwen3.5-2B | `c + a*(1 - exp(-(x/tau)^beta))` | c=0.0956±0.0062, a=0.789±0.017, tau=19.7±0.12, beta=8.45±0.49 | 0.9996 | 2.0 | depth gain, t=5 |
| relresp_Qwen3.5-2B_t5 | Qwen3.5-2B | `c + a*(x-x0)^b` | c=-0.00921±10, a=0.0384±10, b=0.00656±1.9, x0=16±1.5 | 0.8069 | 0.7 | relative response, t=5 |
| hprofile_Qwen3.5-2B_t5 | Qwen3.5-2B | `a*exp(b*x)` | a=1.12±0.27, b=0.144±0.017 | 0.9890 | 16.8 | state norm, t=5 |
| depthgain_Qwen3.5-2B_t50 | Qwen3.5-2B | `c + a*(1 - exp(-(x/tau)^beta))` | c=0.0562±0.0047, a=0.874±0.1, tau=22.1±0.83, beta=5.46±0.39 | 0.9997 | 1.2 | depth gain, t=50 |
| relresp_Qwen3.5-2B_t50 | Qwen3.5-2B | `c + a*(1-exp(-x/t1))*exp(-x/t2)` | c=18.1±2.1, a=-18.1±2.1, t1=0.458±0.015, t2=3.59e+04±0.0021 | 0.9784 | 0.6 | relative response, t=50 |
| hprofile_Qwen3.5-2B_t50 | Qwen3.5-2B | `a*exp(b*x)` | a=1.15±0.28, b=0.147±0.017 | 0.9830 | 19.8 | state norm, t=50 |
| a_t_L27_r0.1 | DeepSeek-1.5B | `a + b*x + c*x^2 + d*x^3` | a=0.179±7.6e-06, b=0.00601±0.00031, c=-5.77e-05±3.2e-06, d=1.23e-07±7.8e-09 | 0.6745 | 12.2 | response, r=0.1 |
| lam10_t_L27 | DeepSeek-1.5B | `c + a*(1 - exp(-(x/tau)^beta))` | c=0.0116±0.0017, a=-0.0252±0.0019, tau=81.9±2.2, beta=20±14 | 0.4041 | 0.0 | finite-time rate |
| fsle_rho1.5 | DeepSeek-1.5B | `c + a*(1-exp(-x/t1))*exp(-x/t2)` | c=0.0259±0.0028, a=0.0293±0.0058, t1=0.00159±0.00099, t2=0.28±0.16 | 0.7195 | 1.3 | growth threshold rho=1.5 |
| fsle_cens_rho1.5 | DeepSeek-1.5B | `a + b*x` | a=0±0, b=0±0 | nan | nan | rho=1.5 |
| fsle_rho2.0 | DeepSeek-1.5B | `a + b*x` | a=0.0433±0, b=0±0 | nan | 11409.3 | growth threshold rho=2.0 |
| fsle_cens_rho2.0 | DeepSeek-1.5B | `a + b*x` | a=0±0, b=0±0 | nan | nan | rho=2.0 |
| fsle_rho4.0 | DeepSeek-1.5B | `c + a*exp(-(x/tau)^2)` | c=0.0243±0.00035, a=-0.00192±0.00061, tau=0.000341±0.00025 | 0.4799 | 0.9 | growth threshold rho=4.0 |
| fsle_cens_rho4.0 | DeepSeek-1.5B | `c + a*x^b/(k^b + x^b)` | c=1.03±0.048, a=-0.97±0.055, b=1.53±0.31, k=2.74e-05±4.1e-06 | 0.9790 | 0.3 | rho=4.0 |
| rstar_ccdf | DeepSeek-1.5B | `c + a*(1-exp(-x/t1))*exp(-x/t2)` | c=0.0512±0.04, a=1.72±0.18, t1=0.109±0.026, t2=0.533±0.076 | 0.9800 | 0.6 | flux-tube radius, r=0.1, 20 directions |
| rstar_hist_x | DeepSeek-1.5B | `c + a*x^b/(k^b + x^b)` | c=0.0169±0.047, a=0.97±0.076, b=2.77±0.41, k=0.664±0.033 | 0.9793 | 3.8 | ECDF |