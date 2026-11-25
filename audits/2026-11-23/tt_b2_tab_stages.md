| stage | hit/helps | non-identity picks when nothing helps | of which worse than identity by >0.5% | worse by >2% | mean pick/identity RMSE | worst | regret |
|---|---|---|---|---|---|---|---|
| LL argmax (Jacobian, YJ near 1 collapsed) | 69/93 | 89/103 | 40 | 12 | 0.9706 | 3.211 | 1.0328 |
| LL + sign-flip null | 59/93 | 53/103 | 26 | 12 | 0.9736 | 3.211 | 1.0362 |
| LL + null + held-out gate (wave-1 recipe) | 57/93 | 17/103 | 0 | 0 | 0.9499 | 1.000 | 1.0123 |
| same with paired sign-flip gate | 54/93 | 10/103 | 0 | 0 | 0.9512 | 1.000 | 1.0137 |
| Huber-robust LL + null + paired gate | 60/93 | 22/103 | 0 | 0 | 0.9501 | 1.000 | 1.0125 |
| extended set (+log1p, sqrt) + paired gate | 56/93 | 12/103 | 0 | 0 | 0.9504 | 1.000 | 1.0136 |
| W1 + support abstention | 48/93 | 8/103 | 0 | 0 | 0.9516 | 1.000 | 1.0140 |
| cross-fitted original-scale max-T (CV) | 87/93 | 31/103 | 0 | 0 | 0.9415 | 1.000 | 1.0032 |
| CV, else W1 | 87/93 | 32/103 | 0 | 0 | 0.9414 | 1.000 | 1.0031 |
(oracle mean ratio 0.9387; 196 runs = 28 cells x 7 seeds)
