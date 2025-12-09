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