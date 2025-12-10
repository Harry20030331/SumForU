# Categories Analysis

Processing category: All_Beauty (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          0.9332  0.9296  0.9333  0.9337  0.8990  0.8085  0.9654  1.0000  6.3209  0.5875  0.0504    0.9859    0.7785    0.7379
baseline    0.1314  0.0070  0.0888  0.0892  0.0000  0.7465  0.9081  1.0000  6.0529  0.3714  0.2444    0.7114    0.7628    0.8213
pe          0.1381  0.0097  0.0885  0.0886  0.0000  0.7330  0.9073  1.0000  6.1142  0.3768  0.1942    0.7140    0.7610    0.8120
sft         0.1408  0.0103  0.0933  0.0936  0.0000  0.7051  0.8897  1.0000  6.1882  0.3143  0.1924    0.7188    0.7525    0.8235
rl          0.1443  0.0084  0.0894  0.0896  0.0000  0.7100  0.8935  1.0000  6.3575  0.2876  0.1904    0.7214    0.7494    0.8336

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.5300  3.5200    0.3156    0.3228     0.0800       0.5600     0.0951         0.2686
pe          1.3300  2.6200    0.4417    0.4282     0.1600       0.6300     0.1591         0.3733
sft         1.1250  1.9625    0.6135    0.5978     0.1600       0.7900     0.1755         0.2984
rl          0.9600  1.8000    0.6807    0.6417     0.2700       0.8300     0.2389         0.3465

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.440        0.347       0.497       0.428
pe              0.420        0.437       0.467       0.441
sft             0.500        0.530       0.437       0.489
rl              0.640        0.687       0.600       0.642

Processing category: Amazon_Fashion (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          0.9219  0.9169  0.9215  0.9219  0.8752  0.8012  0.9534  1.0000  6.1920  0.5530  0.0395    0.9837    0.7737    0.7324
baseline    0.1203  0.0065  0.0843  0.0845  0.0000  0.7021  0.8915  1.0000  5.9864  0.3270  0.2675    0.7092    0.7551    0.8245
pe          0.1169  0.0073  0.0820  0.0822  0.0028  0.6837  0.8837  1.0000  5.9700  0.3304  0.2182    0.7089    0.7533    0.8146
sft         0.1275  0.0068  0.0871  0.0871  0.0000  0.6719  0.8730  1.0000  6.0964  0.2932  0.1805    0.7158    0.7510    0.8182
rl          0.1302  0.0088  0.0830  0.0830  0.0000  0.6584  0.8663  1.0000  6.2090  0.2645  0.1906    0.7199    0.7473    0.8308

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.2700  2.4550    0.5133    0.4947     0.1300       0.6700     0.1476         0.3414
pe          1.2400  2.1650    0.5446    0.5195     0.1900       0.7000     0.1889         0.2707
sft         1.1300  2.1050    0.5654    0.5272     0.2300       0.7700     0.2484         0.3939
rl          1.1000  2.4450    0.5534    0.5234     0.3300       0.7200     0.2909         0.4165

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.470        0.383       0.483       0.446
pe              0.400        0.447       0.460       0.436
sft             0.497        0.503       0.473       0.491
rl              0.633        0.667       0.583       0.628

Processing category: Appliances (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          0.9445  0.9419  0.9446  0.9448  0.9249  0.7952  0.9716  1.0000  6.4327  0.5839  0.0482    0.9883    0.7761    0.7411
baseline    0.1475  0.0120  0.0970  0.0972  0.0000  0.7476  0.9140  1.0000  5.9954  0.3889  0.2612    0.7157    0.7575    0.8237
pe          0.1447  0.0081  0.0941  0.0942  0.0000  0.7001  0.8965  1.0000  6.0043  0.4032  0.1976    0.7177    0.7598    0.8141
sft         0.1398  0.0089  0.0899  0.0905  0.0000  0.6775  0.8789  1.0000  6.1002  0.3194  0.1834    0.7197    0.7509    0.8255
rl          0.1482  0.0097  0.0933  0.0935  0.0000  0.6762  0.8743  1.0000  6.2550  0.2982  0.1862    0.7242    0.7467    0.8376

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.2950  2.6175    0.4266    0.4106     0.1200       0.6600     0.1913         0.2675
pe          1.2550  2.3275    0.4349    0.4022     0.2000       0.6500     0.2287         0.3247
sft         1.1050  1.8875    0.5769    0.5477     0.2300       0.7900     0.2060         0.2857
rl          1.1600  2.2200    0.5630    0.5438     0.1700       0.7600     0.1597         0.2698

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.433        0.310       0.537       0.427
pe              0.423        0.503       0.427       0.451
sft             0.517        0.533       0.463       0.504
rl              0.627        0.653       0.573       0.618

Processing category: Baby_Products (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          0.9462  0.9434  0.9462  0.9460  0.9218  0.7796  0.9477  1.0000  6.4011  0.5992  0.0454    0.9881    0.7783    0.7319
baseline    0.1372  0.0107  0.0931  0.0927  0.0000  0.7658  0.9258  1.0000  6.1290  0.4036  0.2198    0.7043    0.7604    0.8146
pe          0.1511  0.0102  0.0966  0.0963  0.0000  0.7181  0.8969  1.0000  6.1116  0.4030  0.1934    0.7094    0.7593    0.8153
sft         0.1469  0.0118  0.0933  0.0930  0.0000  0.6915  0.8772  1.0000  6.1948  0.3320  0.1742    0.7126    0.7509    0.8209
rl          0.1538  0.0107  0.0964  0.0964  0.0041  0.6924  0.8799  1.0000  6.3181  0.3176  0.1803    0.7168    0.7505    0.8299

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.4600  3.2200    0.3591    0.3510     0.1300       0.5800     0.1334         0.2959
pe          1.3350  2.4625    0.4285    0.4044     0.1600       0.6200     0.1469         0.2244
sft         1.1900  2.2550    0.5036    0.4691     0.2300       0.7100     0.1970         0.2838
rl          1.1500  2.4650    0.5183    0.5210     0.2400       0.7400     0.2042         0.2983

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.397        0.300       0.407       0.368
pe              0.413        0.447       0.450       0.437
sft             0.530        0.557       0.490       0.525
rl              0.660        0.697       0.653       0.670

Processing category: CDs_and_Vinyl (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          0.9299  0.9260  0.9309  0.9302  0.9159  0.8455  0.9704  1.0000  6.6069  0.5814  0.0493    0.9877    0.7682    0.7374
baseline    0.1317  0.0141  0.0883  0.0883  0.0040  0.7209  0.9019  1.0000  5.9669  0.3812  0.3034    0.7123    0.7765    0.8216
pe          0.1335  0.0111  0.0894  0.0894  0.0043  0.6992  0.8938  1.0000  6.0254  0.3744  0.2569    0.7134    0.7731    0.8163
sft         0.1387  0.0135  0.0914  0.0913  0.0066  0.6985  0.8870  1.0000  6.1563  0.3305  0.1920    0.7181    0.7716    0.8122
rl          0.1434  0.0130  0.0899  0.0902  0.0052  0.6970  0.8921  1.0000  6.3379  0.3069  0.1930    0.7235    0.7674    0.8263

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.0250  1.9025    0.0837    0.0776     0.0900       0.8100     0.0356         0.2000
pe          1.0750  1.9725    0.0858    0.1120     0.0900       0.8000     0.0553         0.1985
sft         0.9500  1.6000    0.3391    0.3170     0.1500       0.8300     0.1353         0.2688
rl          0.8400  1.4000    0.4779    0.3726     0.2200       0.8500     0.1857         0.2361

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.563        0.400       0.560       0.508
pe              0.450        0.467       0.437       0.451
sft             0.423        0.520       0.403       0.449
rl              0.563        0.613       0.600       0.592

Processing category: Gift_Cards (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          0.9235  0.9190  0.9235  0.9229  0.8579  0.7819  0.9486  0.9900  6.0009  0.5718  0.0955    0.9865    0.7666    0.7508
baseline    0.1797  0.0225  0.1211  0.1208  0.0000  0.6711  0.8546  1.0000  5.7940  0.3413  0.2801    0.7315    0.7563    0.8415
pe          0.1650  0.0211  0.1127  0.1123  0.0000  0.6303  0.8434  1.0000  5.7617  0.3337  0.2219    0.7269    0.7533    0.8311
sft         0.1556  0.0179  0.1027  0.1025  0.0030  0.6189  0.8265  1.0000  5.9150  0.2730  0.2029    0.7282    0.7467    0.8383
rl          0.1570  0.0168  0.1020  0.1018  0.0000  0.6220  0.8333  1.0000  6.0525  0.2676  0.2008    0.7320    0.7443    0.8466

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.1950  2.4025    0.4705    0.4368     0.0700       0.7000     0.1755         0.2824
pe          1.6150  3.5325    0.4402    0.4059     0.1000       0.4900     0.1668         0.2941
sft         1.4200  3.0550    0.5736    0.5613     0.1400       0.6300     0.1918         0.3294
rl          1.5500  4.0250    0.4924    0.5055     0.1600       0.5900     0.1620         0.2621

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.457        0.327       0.567       0.450
pe              0.390        0.477       0.350       0.406
sft             0.510        0.517       0.460       0.496
rl              0.643        0.680       0.623       0.649

Processing category: Handmade_Products (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          0.9275  0.9234  0.9280  0.9280  0.8924  0.8104  0.9722  1.0000  6.2421  0.5511  0.0596    0.9867    0.7762    0.7431
baseline    0.1454  0.0085  0.1005  0.1008  0.0000  0.7230  0.8961  1.0000  5.9682  0.3150  0.2990    0.7154    0.7604    0.8302
pe          0.1480  0.0078  0.0979  0.0983  0.0000  0.6928  0.8824  1.0000  6.0110  0.3216  0.2197    0.7185    0.7596    0.8203
sft         0.1554  0.0084  0.1027  0.1026  0.0000  0.6704  0.8586  1.0000  6.0907  0.2827  0.2073    0.7227    0.7539    0.8274
rl          0.1583  0.0107  0.0983  0.0984  0.0000  0.6670  0.8601  1.0000  6.2530  0.2550  0.2008    0.7268    0.7513    0.8407

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.0606  2.1263    0.2955    0.3661     0.0808       0.7879     0.0568         0.1864
pe          1.0600  1.6050    0.5592    0.5222     0.0800       0.8200     0.1240         0.2663
sft         0.9150  1.3025    0.6700    0.5996     0.0900       0.8900     0.1177         0.2160
rl          0.7400  1.0800    0.7521    0.6625     0.2300       0.8800     0.2773         0.3889

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.470        0.373       0.460       0.435
pe              0.423        0.433       0.423       0.427
sft             0.440        0.547       0.437       0.474
rl              0.667        0.647       0.680       0.664

Processing category: Health_and_Personal_Care (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          0.9354  0.9312  0.9348  0.9355  0.9174  0.8226  0.9739  1.0000  6.5395  0.5690  0.0560    0.9876    0.7710    0.7369
baseline    0.1273  0.0058  0.0895  0.0890  0.0000  0.7981  0.9408  1.0000  6.2363  0.4019  0.2214    0.7037    0.7653    0.8127
pe          0.1382  0.0056  0.0884  0.0881  0.0036  0.7527  0.9151  1.0000  6.1885  0.4055  0.1877    0.7071    0.7633    0.8075
sft         0.1474  0.0096  0.0898  0.0897  0.0037  0.7297  0.8976  1.0000  6.2669  0.3242  0.1946    0.7125    0.7537    0.8217
rl          0.1409  0.0095  0.0890  0.0890  0.0035  0.7261  0.9001  1.0000  6.4142  0.2966  0.1888    0.7141    0.7513    0.8304

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.3200  2.8800    0.4055    0.4446     0.1100       0.6600     0.1040         0.1744
pe          1.2300  2.1600    0.5345    0.5370     0.1200       0.7000     0.1395         0.1944
sft         1.0700  1.8700    0.6210    0.6385     0.1800       0.7700     0.2042         0.2781
rl          0.9850  2.0375    0.6139    0.6252     0.2900       0.7800     0.2633         0.3300

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.413        0.373       0.410       0.399
pe              0.417        0.477       0.443       0.446
sft             0.523        0.510       0.520       0.518
rl              0.647        0.640       0.627       0.638

Processing category: Magazine_Subscriptions (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          0.9301  0.9265  0.9303  0.9300  0.8864  0.8089  0.9585  1.0000  6.2905  0.5404  0.0620    0.9877    0.7699    0.7466
baseline    0.1341  0.0077  0.0876  0.0873  0.0000  0.7467  0.9151  1.0000  6.0938  0.3512  0.2867    0.7170    0.7678    0.8388
pe          0.1315  0.0083  0.0902  0.0901  0.0000  0.6927  0.8947  1.0000  5.9933  0.3541  0.2212    0.7173    0.7656    0.8243
sft         0.1392  0.0087  0.0924  0.0923  0.0000  0.6627  0.8660  1.0000  6.0914  0.3126  0.2109    0.7220    0.7601    0.8295
rl          0.1356  0.0078  0.0871  0.0870  0.0027  0.6801  0.8828  1.0000  6.2876  0.2942  0.2015    0.7244    0.7594    0.8379

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.1000  2.0000    0.2751    0.2612     0.1500       0.8000     0.1406         0.2972
pe          1.2300  2.1300    0.3592    0.2947     0.1700       0.7200     0.2343         0.3436
sft         1.1550  1.9475    0.4718    0.4279     0.1200       0.7700     0.2322         0.3494
rl          1.1750  2.2625    0.4732    0.4119     0.1700       0.7500     0.2212         0.3464

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.520        0.353       0.597       0.490
pe              0.423        0.430       0.393       0.416
sft             0.497        0.550       0.443       0.497
rl              0.560        0.667       0.567       0.598

Processing category: Software (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          0.9073  0.9010  0.9070  0.9072  0.8209  0.8156  0.9629  1.0000  5.9655  0.5537  0.0565    0.9840    0.7671    0.7304
baseline    0.1229  0.0106  0.0909  0.0905  0.0000  0.7301  0.9097  1.0000  5.9971  0.2898  0.2992    0.7142    0.7563    0.8286
pe          0.1210  0.0084  0.0877  0.0876  0.0000  0.6910  0.8825  1.0000  5.9367  0.3081  0.2376    0.7127    0.7580    0.8162
sft         0.1137  0.0074  0.0842  0.0842  0.0000  0.6782  0.8794  1.0000  6.0576  0.2471  0.2154    0.7149    0.7511    0.8212
rl          0.1126  0.0079  0.0805  0.0803  0.0000  0.6706  0.8742  1.0000  6.1595  0.2279  0.2115    0.7169    0.7481    0.8312

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.1050  1.9775    0.5048    0.5079     0.1900       0.7800     0.1435         0.2125
pe          1.1450  1.9425    0.5091    0.5100     0.2300       0.7600     0.2045         0.2871
sft         1.0700  1.8000    0.5531    0.5607     0.2200       0.7700     0.1836         0.2390
rl          1.1200  2.2500    0.5014    0.5107     0.2200       0.7400     0.2194         0.2875

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.500        0.413       0.480       0.465
pe              0.427        0.550       0.420       0.466
sft             0.457        0.453       0.500       0.470
rl              0.617        0.583       0.600       0.600

=== Overall LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.467        0.358       0.525       0.450
pe              0.420        0.489       0.396       0.435
sft             0.479        0.534       0.460       0.491
rl              0.634        0.619       0.618       0.624

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.441        0.347       0.497       0.428
pe              0.420        0.436       0.396       0.417
sft             0.456        0.453       0.433       0.447
rl              0.617        0.687       0.600       0.635


# Comparison Analysis
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          1.0000  1.0000  1.0000  1.0000  1.0000  0.6697  0.9247  0.9990  7.0896  0.5870  0.0700    1.0000    0.7828    0.7283
baseline    0.1379  0.0105  0.0941  0.0941  0.0018  0.5911  0.8391  1.0000  6.8543  0.3571  0.2683    0.7135    0.7618    0.8257
pe          0.1390  0.0098  0.0927  0.0927  0.0028  0.5403  0.8032  1.0000  6.7607  0.3611  0.2148    0.7146    0.7606    0.8172
sft         0.1408  0.0104  0.0926  0.0926  0.0033  0.5180  0.7837  1.0000  6.8061  0.3029  0.1954    0.7185    0.7542    0.8238
rl          0.1427  0.0104  0.0909  0.0909  0.0028  0.5155  0.7862  1.0000  6.9287  0.2816  0.1944    0.7220    0.7516    0.8345

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.2362  2.5105    0.4219    0.4233     0.1151       0.7007     0.1310         0.2484
pe          1.2515  2.2917    0.4709    0.4498     0.1500       0.6890     0.1627         0.2810
sft         1.1130  1.9785    0.5772    0.5583     0.1750       0.7720     0.1821         0.2949
rl          1.0780  2.1985    0.5847    0.5629     0.2300       0.7640     0.2334         0.3142

JUEGE: "openai/gpt-oss-120b"
--- METHOD: BASELINE ---
Overall Average Score: 0.441
Total Wins: 3966 / 9000
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.464 |
| Grounding       | 0.361 |
| Persona         | 0.497 |

--- METHOD: PE ---
Overall Average Score: 0.439
Total Wins: 3948 / 9000
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.419 |
| Grounding       | 0.466 |
| Persona         | 0.431 |

--- METHOD: SFT ---
Overall Average Score: 0.490
Total Wins: 4413 / 9000
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.488 |
| Grounding       | 0.532 |
| Persona         | 0.452 |

--- METHOD: RL ---
Overall Average Score: 0.630
Total Wins: 5673 / 9000
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.629 |
| Grounding       | 0.642 |
| Persona         | 0.620 |

JUDGE: "Qwen/Qwen3-235B-A22B-Instruct-2507"
--- METHOD: BASELINE ---
Overall Average Score: 0.336
Total Wins: 3022 / 9000
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.441 |
| Grounding       | 0.331 |
| Persona         | 0.235 |

--- METHOD: PE ---
Overall Average Score: 0.489
Total Wins: 4400 / 9000
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.435 |
| Grounding       | 0.709 |
| Persona         | 0.322 |

--- METHOD: SFT ---
Overall Average Score: 0.507
Total Wins: 4566 / 9000
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.440 |
| Grounding       | 0.531 |
| Persona         | 0.551 |

--- METHOD: RL ---
Overall Average Score: 0.668
Total Wins: 6012 / 9000
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.684 |
| Grounding       | 0.429 |
| Persona         | 0.892 |

Processing category: Video_Games (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          1.0000  1.0000  1.0000  1.0000  1.0000  0.7923  0.9659  1.0000  6.7150  0.5957  0.0538    1.0000    0.7839    0.7324
baseline    0.1245  0.0098  0.0857  0.0861  0.0000  0.7579  0.9298  1.0000  6.1231  0.4018  0.2742    0.6980    0.7682    0.8182
pe          0.1401  0.0109  0.0925  0.0930  0.0010  0.7348  0.9133  1.0000  6.1060  0.4193  0.2104    0.7007    0.7674    0.8111
sft         0.1449  0.0098  0.0960  0.0961  0.0013  0.7283  0.9171  1.0000  6.2431  0.3418  0.2091    0.7067    0.7585    0.8223
rl          0.1552  0.0113  0.0980  0.0982  0.0024  0.7209  0.9046  1.0000  6.3939  0.3191  0.1947    0.7104    0.7557    0.8307

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.1300  2.1300    0.5291    0.5199     0.1900       0.7200     0.1992         0.3038
pe          1.0700  1.7700    0.5739    0.5720     0.2100       0.7900     0.2027         0.2886
sft         0.9450  1.5075    0.6564    0.6602     0.2300       0.8200     0.1876         0.3130
rl          1.0500  2.1800    0.5459    0.5522     0.2800       0.7800     0.2274         0.3105

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.443        0.360       0.440       0.414
pe              0.483        0.610       0.420       0.504
sft             0.443        0.400       0.503       0.449
rl              0.630        0.630       0.637       0.632

Processing category: Arts_Crafts_and_Sewing (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          1.0000  1.0000  1.0000  1.0000  1.0000  0.8240  0.9732  1.0000  6.3948  0.5676  0.0640    1.0000    0.7788    0.7252
baseline    0.1336  0.0083  0.0918  0.0915  0.0034  0.7528  0.9153  1.0000  6.0597  0.3917  0.2473    0.7201    0.7664    0.8186
pe          0.1403  0.0094  0.0952  0.0951  0.0000  0.7402  0.9101  1.0000  6.1186  0.4160  0.1883    0.7223    0.7662    0.8136
sft         0.1395  0.0076  0.0923  0.0922  0.0036  0.7003  0.8858  1.0000  6.1633  0.3414  0.1870    0.7249    0.7571    0.8236
rl          0.1399  0.0068  0.0905  0.0903  0.0033  0.7042  0.8864  1.0000  6.3472  0.2995  0.1886    0.7269    0.7507    0.8338

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.3550  3.0075    0.3113    0.3184     0.1500       0.6400     0.1663         0.2746
pe          1.3050  2.5375    0.3944    0.3769     0.2100       0.6800     0.2032         0.3527
sft         1.1900  2.2900    0.4996    0.4884     0.2100       0.7300     0.1927         0.3239
rl          1.1600  2.4350    0.5511    0.5335     0.2100       0.7500     0.2080         0.2743

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.480        0.390       0.457       0.442
pe              0.490        0.610       0.370       0.490
sft             0.390        0.413       0.557       0.453
rl              0.640        0.587       0.617       0.614

Processing category: Industrial_and_Scientific (samples: 100)
=== Text quality, diversity, semantic similarity, and coverage ===
method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          1.0000  1.0000  1.0000  1.0000  1.0000  0.8281  0.9728  1.0000  6.5760  0.5652  0.0547    1.0000    0.7812    0.7243
baseline    0.1287  0.0082  0.0892  0.0893  0.0000  0.7821  0.9347  1.0000  6.1991  0.3892  0.2539    0.7059    0.7591    0.8190
pe          0.1371  0.0071  0.0887  0.0887  0.0030  0.7527  0.9197  1.0000  6.1515  0.3906  0.2051    0.7082    0.7584    0.8124
sft         0.1411  0.0067  0.0888  0.0888  0.0034  0.7153  0.8927  1.0000  6.2390  0.3229  0.1928    0.7128    0.7496    0.8232
rl          0.1452  0.0073  0.0891  0.0891  0.0000  0.7134  0.8915  1.0000  6.4038  0.3023  0.1893    0.7151    0.7460    0.8322

=== Suitability vs reference rating (0-5 scale) ===
method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
baseline    1.3100  2.8300    0.4799    0.4882     0.1200       0.6900     0.1463         0.2445
pe          1.1900  2.2250    0.5204    0.5058     0.1400       0.7200     0.1503         0.2765
sft         1.0650  2.0725    0.5715    0.5630     0.2500       0.7700     0.2340         0.3722
rl          1.0600  2.4500    0.5419    0.5418     0.2900       0.7700     0.2729         0.3678

=== LLM Judge Three-Dimensional Comparison ===
method          Consistency  Grounding   Persona     Overall
baseline        0.500        0.310       0.463       0.424
pe              0.460        0.597       0.457       0.504
sft             0.417        0.410       0.473       0.433
rl              0.623        0.683       0.607       0.638

# Old Results about All Beauty
=== Text quality, diversity, semantic similarity, and coverage ===

method          r1      r2      rL   rLsum   bleu4     D-2     D-3     USR    ENTR  RevCov PersCov   RefBS-R   RevBS-P  PersBS-R
gt          1.0000  1.0000  1.0000  1.0000  1.0000  0.8271  0.9728  1.0000  6.3820  0.6996  0.0625    1.0000    0.8428    0.8321
235b_ref    0.1610  0.0106  0.0979  0.0977  0.0000  0.7228  0.9053  1.0000  6.2516  0.3947  0.1839    0.8406    0.8416    0.8760
baseline    0.1509  0.0091  0.0977  0.0975  0.0000  0.7725  0.9259  1.0000  6.1601  0.4618  0.2380    0.8367    0.8509    0.8746
pe          0.1528  0.0094  0.0957  0.0956  0.0000  0.7430  0.9140  1.0000  6.2274  0.4759  0.1883    0.8389    0.8495    0.8733
sft         0.1486  0.0080  0.0919  0.0919  0.0000  0.7125  0.8941  1.0000  6.2076  0.4136  0.1768    0.8377    0.8424    0.8738
rl          0.1494  0.0082  0.0867  0.0864  0.0000  0.7089  0.8986  1.0000  6.5932  0.4145  0.1485    0.8383    0.8322    0.8765

BERTScore F1 per method (summary vs reference):
  gt         BERTScore-F1 = 1.0000
  235b_ref   BERTScore-F1 = 0.8381
  baseline   BERTScore-F1 = 0.8390
  pe         BERTScore-F1 = 0.8390
  sft        BERTScore-F1 = 0.8359
  rl         BERTScore-F1 = 0.8271

=== Suitability vs reference rating (0-5 scale) ===

method         MAE     MSE   Pearson  Spearman   ExactAcc   Within1Acc    MacroF1    BalancedAcc
gt          0.0000  0.0000    1.0000    1.0000     1.0000       1.0000     1.0000         1.0000
235b_ref    1.1150  1.8875    0.6431    0.6571     0.1700       0.7700     0.1652         0.3003
baseline    1.6050  3.8225    0.2788    0.2294     0.0700       0.5900     0.0788         0.1737
pe          1.4700  3.0850    0.3631    0.3764     0.1000       0.5900     0.1027         0.2097
sft         1.3700  2.7750    0.4222    0.4230     0.1400       0.6200     0.1266         0.2394
rl          1.3650  3.4075    0.4193    0.4489     0.1500       0.6900     0.1476         0.2404

JUDGE: "Qwen/Qwen3-235B-A22B-Instruct-2507"
--- METHOD: BASELINE ---
Overall Average Score: 0.433
Total Wins: 390 / 900
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.503 |
| Grounding       | 0.337 |
| Persona         | 0.460 |

--- METHOD: PE ---
Overall Average Score: 0.432
Total Wins: 389 / 900
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.403 |
| Grounding       | 0.463 |
| Persona         | 0.430 |

--- METHOD: SFT ---
Overall Average Score: 0.450
Total Wins: 405 / 900
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.393 |
| Grounding       | 0.520 |
| Persona         | 0.437 |

--- METHOD: RL ---
Overall Average Score: 0.684
Total Wins: 616 / 900
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.700 |
| Grounding       | 0.680 |
| Persona         | 0.673 |

JUDGE: "openai/gpt-oss-120b"
--- METHOD: BASELINE ---
Overall Average Score: 0.433
Total Wins: 390 / 900
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.503 |
| Grounding       | 0.337 |
| Persona         | 0.460 |

--- METHOD: PE ---
Overall Average Score: 0.432
Total Wins: 389 / 900
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.403 |
| Grounding       | 0.463 |
| Persona         | 0.430 |

--- METHOD: SFT ---
Overall Average Score: 0.450
Total Wins: 405 / 900
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.393 |
| Grounding       | 0.520 |
| Persona         | 0.437 |

--- METHOD: RL ---
Overall Average Score: 0.684
Total Wins: 616 / 900
| Dimension       | Win Rate |
|---------------|---------|
| Consistency     | 0.700 |
| Grounding       | 0.680 |
| Persona         | 0.673 |

# User Study Results
## Data Indices

The following table lists the selected data indices for each category:

| Category | Source File | Line Index | Global Index |
|----------|-------------|------------|--------------|
| All_Beauty | All_Beauty.jsonl | 81 | 81 |
| Amazon_Fashion | Amazon_Fashion.jsonl | 14 | 114 |
| Appliances | Appliances.jsonl | 3 | 203 |
| Baby_Products | Baby_Products.jsonl | 94 | 394 |
| CDs_and_Vinyl | CDs_and_Vinyl.jsonl | 35 | 435 |
| Gift_Cards | Gift_Cards.jsonl | 31 | 531 |
| Handmade_Products | Handmade_Products.jsonl | 28 | 628 |
| Health_and_Personal_Care | Health_and_Personal_Care.jsonl | 17 | 717 |
| Magazine_Subscriptions | Magazine_Subscriptions.jsonl | 94 | 894 |
| Software | Software.jsonl | 13 | 913 |