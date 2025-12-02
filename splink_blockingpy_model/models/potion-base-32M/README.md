---
library_name: model2vec
license: mit
model-index:
- name: potion-base-32M
  results:
  - dataset:
      config: en-ext
      name: MTEB AmazonCounterfactualClassification (en-ext)
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
      split: test
      type: mteb/amazon_counterfactual
    metrics:
    - type: accuracy
      value: 74.49025487256372
    - type: ap
      value: 23.053406998271548
    - type: ap_weighted
      value: 23.053406998271548
    - type: f1
      value: 61.61224310463791
    - type: f1_weighted
      value: 79.15713131602897
    - type: main_score
      value: 74.49025487256372
    task:
      type: Classification
  - dataset:
      config: en
      name: MTEB AmazonCounterfactualClassification (en)
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
      split: test
      type: mteb/amazon_counterfactual
    metrics:
    - type: accuracy
      value: 74.55223880597013
    - type: ap
      value: 36.777904971672484
    - type: ap_weighted
      value: 36.777904971672484
    - type: f1
      value: 68.20927320328308
    - type: f1_weighted
      value: 76.8028646180125
    - type: main_score
      value: 74.55223880597013
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB AmazonPolarityClassification (default)
      revision: e2d317d38cd51312af73b3d32a06d1a08b442046
      split: test
      type: mteb/amazon_polarity
    metrics:
    - type: accuracy
      value: 72.855975
    - type: ap
      value: 67.07977033292134
    - type: ap_weighted
      value: 67.07977033292134
    - type: f1
      value: 72.67632985018474
    - type: f1_weighted
      value: 72.67632985018474
    - type: main_score
      value: 72.855975
    task:
      type: Classification
  - dataset:
      config: en
      name: MTEB AmazonReviewsClassification (en)
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
      split: test
      type: mteb/amazon_reviews_multi
    metrics:
    - type: accuracy
      value: 36.948
    - type: f1
      value: 36.39230651926405
    - type: f1_weighted
      value: 36.39230651926405
    - type: main_score
      value: 36.948
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB ArguAna (default)
      revision: c22ab2a51041ffd869aaddef7af8d8215647e41a
      split: test
      type: mteb/arguana
    metrics:
    - type: main_score
      value: 42.864000000000004
    - type: map_at_1
      value: 21.693
    - type: map_at_10
      value: 34.859
    - type: map_at_100
      value: 36.014
    - type: map_at_1000
      value: 36.047000000000004
    - type: map_at_20
      value: 35.667
    - type: map_at_3
      value: 30.416999999999998
    - type: map_at_5
      value: 32.736
    - type: mrr_at_1
      value: 22.04836415362731
    - type: mrr_at_10
      value: 35.01442231705384
    - type: mrr_at_100
      value: 36.16267051020847
    - type: mrr_at_1000
      value: 36.19625564960624
    - type: mrr_at_20
      value: 35.81309792569356
    - type: mrr_at_3
      value: 30.547652916073904
    - type: mrr_at_5
      value: 32.87339971550487
    - type: nauc_map_at_1000_diff1
      value: 7.561195580746018
    - type: nauc_map_at_1000_max
      value: -1.556531946821957
    - type: nauc_map_at_1000_std
      value: 2.056871021244521
    - type: nauc_map_at_100_diff1
      value: 7.576648616531427
    - type: nauc_map_at_100_max
      value: -1.5197684321804203
    - type: nauc_map_at_100_std
      value: 2.102558505658414
    - type: nauc_map_at_10_diff1
      value: 7.643409260188448
    - type: nauc_map_at_10_max
      value: -1.5534104754693818
    - type: nauc_map_at_10_std
      value: 1.8735258045916798
    - type: nauc_map_at_1_diff1
      value: 8.370318054971092
    - type: nauc_map_at_1_max
      value: -5.083984587735291
    - type: nauc_map_at_1_std
      value: -1.8039233134026431
    - type: nauc_map_at_20_diff1
      value: 7.642516551976743
    - type: nauc_map_at_20_max
      value: -1.388835890563647
    - type: nauc_map_at_20_std
      value: 2.198921728682202
    - type: nauc_map_at_3_diff1
      value: 7.437604281774142
    - type: nauc_map_at_3_max
      value: -2.7586587623340932
    - type: nauc_map_at_3_std
      value: 0.8031910070187186
    - type: nauc_map_at_5_diff1
      value: 6.80651166389857
    - type: nauc_map_at_5_max
      value: -2.7399645587571806
    - type: nauc_map_at_5_std
      value: 1.0580951572345365
    - type: nauc_mrr_at_1000_diff1
      value: 6.27575281564605
    - type: nauc_mrr_at_1000_max
      value: -2.0467879398352458
    - type: nauc_mrr_at_1000_std
      value: 1.9897114385666632
    - type: nauc_mrr_at_100_diff1
      value: 6.292566922480118
    - type: nauc_mrr_at_100_max
      value: -2.009602726575689
    - type: nauc_mrr_at_100_std
      value: 2.0353272285661115
    - type: nauc_mrr_at_10_diff1
      value: 6.38514525903419
    - type: nauc_mrr_at_10_max
      value: -2.0386434404188583
    - type: nauc_mrr_at_10_std
      value: 1.7937484255337244
    - type: nauc_mrr_at_1_diff1
      value: 7.131931862611085
    - type: nauc_mrr_at_1_max
      value: -5.008568891508268
    - type: nauc_mrr_at_1_std
      value: -1.86541494834969
    - type: nauc_mrr_at_20_diff1
      value: 6.352383732997516
    - type: nauc_mrr_at_20_max
      value: -1.8916791965400346
    - type: nauc_mrr_at_20_std
      value: 2.142946311516978
    - type: nauc_mrr_at_3_diff1
      value: 5.952701132344548
    - type: nauc_mrr_at_3_max
      value: -3.433767309685429
    - type: nauc_mrr_at_3_std
      value: 0.8212723818638477
    - type: nauc_mrr_at_5_diff1
      value: 5.518638249091068
    - type: nauc_mrr_at_5_max
      value: -3.284414027772663
    - type: nauc_mrr_at_5_std
      value: 0.8740053182401986
    - type: nauc_ndcg_at_1000_diff1
      value: 7.853268129426508
    - type: nauc_ndcg_at_1000_max
      value: 0.07872546898149692
    - type: nauc_ndcg_at_1000_std
      value: 3.830950311415248
    - type: nauc_ndcg_at_100_diff1
      value: 8.18494720374052
    - type: nauc_ndcg_at_100_max
      value: 1.189039585107088
    - type: nauc_ndcg_at_100_std
      value: 5.162437147506563
    - type: nauc_ndcg_at_10_diff1
      value: 8.483384610768821
    - type: nauc_ndcg_at_10_max
      value: 1.2922857488042296
    - type: nauc_ndcg_at_10_std
      value: 4.364359149153261
    - type: nauc_ndcg_at_1_diff1
      value: 8.370318054971092
    - type: nauc_ndcg_at_1_max
      value: -5.083984587735291
    - type: nauc_ndcg_at_1_std
      value: -1.8039233134026431
    - type: nauc_ndcg_at_20_diff1
      value: 8.635794468766242
    - type: nauc_ndcg_at_20_max
      value: 2.142313693153693
    - type: nauc_ndcg_at_20_std
      value: 5.854124318847265
    - type: nauc_ndcg_at_3_diff1
      value: 7.5258085340807375
    - type: nauc_ndcg_at_3_max
      value: -1.835003355061091
    - type: nauc_ndcg_at_3_std
      value: 1.7180856674185805
    - type: nauc_ndcg_at_5_diff1
      value: 6.454885361450212
    - type: nauc_ndcg_at_5_max
      value: -1.7697904754470226
    - type: nauc_ndcg_at_5_std
      value: 2.23730543193386
    - type: nauc_precision_at_1000_diff1
      value: 13.463008420949352
    - type: nauc_precision_at_1000_max
      value: 39.854067665230545
    - type: nauc_precision_at_1000_std
      value: 59.278094323029116
    - type: nauc_precision_at_100_diff1
      value: 17.135034752024826
    - type: nauc_precision_at_100_max
      value: 37.32457612526076
    - type: nauc_precision_at_100_std
      value: 48.881195912340196
    - type: nauc_precision_at_10_diff1
      value: 12.284655559397713
    - type: nauc_precision_at_10_max
      value: 12.655164738763295
    - type: nauc_precision_at_10_std
      value: 14.111055058962119
    - type: nauc_precision_at_1_diff1
      value: 8.370318054971092
    - type: nauc_precision_at_1_max
      value: -5.083984587735291
    - type: nauc_precision_at_1_std
      value: -1.8039233134026431
    - type: nauc_precision_at_20_diff1
      value: 15.208076882937696
    - type: nauc_precision_at_20_max
      value: 22.831763946168888
    - type: nauc_precision_at_20_std
      value: 27.573772369307004
    - type: nauc_precision_at_3_diff1
      value: 7.860638544154737
    - type: nauc_precision_at_3_max
      value: 0.6713212806084865
    - type: nauc_precision_at_3_std
      value: 4.175512987337371
    - type: nauc_precision_at_5_diff1
      value: 5.479186086763304
    - type: nauc_precision_at_5_max
      value: 0.98921018748054
    - type: nauc_precision_at_5_std
      value: 5.630076964069638
    - type: nauc_recall_at_1000_diff1
      value: 13.46300842095073
    - type: nauc_recall_at_1000_max
      value: 39.854067665229756
    - type: nauc_recall_at_1000_std
      value: 59.27809432303065
    - type: nauc_recall_at_100_diff1
      value: 17.135034752024637
    - type: nauc_recall_at_100_max
      value: 37.32457612526039
    - type: nauc_recall_at_100_std
      value: 48.88119591234045
    - type: nauc_recall_at_10_diff1
      value: 12.28465555939771
    - type: nauc_recall_at_10_max
      value: 12.655164738763315
    - type: nauc_recall_at_10_std
      value: 14.111055058962066
    - type: nauc_recall_at_1_diff1
      value: 8.370318054971092
    - type: nauc_recall_at_1_max
      value: -5.083984587735291
    - type: nauc_recall_at_1_std
      value: -1.8039233134026431
    - type: nauc_recall_at_20_diff1
      value: 15.208076882937634
    - type: nauc_recall_at_20_max
      value: 22.83176394616889
    - type: nauc_recall_at_20_std
      value: 27.573772369307076
    - type: nauc_recall_at_3_diff1
      value: 7.860638544154747
    - type: nauc_recall_at_3_max
      value: 0.6713212806084956
    - type: nauc_recall_at_3_std
      value: 4.175512987337308
    - type: nauc_recall_at_5_diff1
      value: 5.479186086763291
    - type: nauc_recall_at_5_max
      value: 0.989210187480526
    - type: nauc_recall_at_5_std
      value: 5.630076964069639
    - type: ndcg_at_1
      value: 21.693
    - type: ndcg_at_10
      value: 42.864000000000004
    - type: ndcg_at_100
      value: 48.22
    - type: ndcg_at_1000
      value: 49.027
    - type: ndcg_at_20
      value: 45.788000000000004
    - type: ndcg_at_3
      value: 33.458
    - type: ndcg_at_5
      value: 37.687
    - type: precision_at_1
      value: 21.693
    - type: precision_at_10
      value: 6.877999999999999
    - type: precision_at_100
      value: 0.932
    - type: precision_at_1000
      value: 0.099
    - type: precision_at_20
      value: 4.015
    - type: precision_at_3
      value: 14.106
    - type: precision_at_5
      value: 10.541
    - type: recall_at_1
      value: 21.693
    - type: recall_at_10
      value: 68.777
    - type: recall_at_100
      value: 93.243
    - type: recall_at_1000
      value: 99.431
    - type: recall_at_20
      value: 80.29899999999999
    - type: recall_at_3
      value: 42.319
    - type: recall_at_5
      value: 52.703
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB ArxivClusteringP2P (default)
      revision: a122ad7f3f0291bf49cc6f4d32aa80929df69d5d
      split: test
      type: mteb/arxiv-clustering-p2p
    metrics:
    - type: main_score
      value: 37.21515684139779
    - type: v_measure
      value: 37.21515684139779
    - type: v_measure_std
      value: 13.948324903262096
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB ArxivClusteringS2S (default)
      revision: f910caf1a6075f7329cdf8c1a6135696f37dbd53
      split: test
      type: mteb/arxiv-clustering-s2s
    metrics:
    - type: main_score
      value: 27.89275646771196
    - type: v_measure
      value: 27.89275646771196
    - type: v_measure_std
      value: 14.54879669291749
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB AskUbuntuDupQuestions (default)
      revision: 2000358ca161889fa9c082cb41daa8dcfb161a54
      split: test
      type: mteb/askubuntudupquestions-reranking
    metrics:
    - type: main_score
      value: 54.23949583322935
    - type: map
      value: 54.23949583322935
    - type: mrr
      value: 67.55825968429846
    - type: nAUC_map_diff1
      value: 15.161467557403707
    - type: nAUC_map_max
      value: 17.924242718354826
    - type: nAUC_map_std
      value: 11.333118592351424
    - type: nAUC_mrr_diff1
      value: 22.993618051206965
    - type: nAUC_mrr_max
      value: 22.90209504491936
    - type: nAUC_mrr_std
      value: 12.131969980175453
    task:
      type: Reranking
  - dataset:
      config: default
      name: MTEB BIOSSES (default)
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
      split: test
      type: mteb/biosses-sts
    metrics:
    - type: cosine_pearson
      value: 79.97535229997727
    - type: cosine_spearman
      value: 77.55658645654347
    - type: euclidean_pearson
      value: 78.45282631461923
    - type: euclidean_spearman
      value: 77.55658645654347
    - type: main_score
      value: 77.55658645654347
    - type: manhattan_pearson
      value: 78.29319221254525
    - type: manhattan_spearman
      value: 76.68849438732013
    - type: pearson
      value: 79.97535229997727
    - type: spearman
      value: 77.55658645654347
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB Banking77Classification (default)
      revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
      split: test
      type: mteb/banking77
    metrics:
    - type: accuracy
      value: 74.78246753246752
    - type: f1
      value: 74.03440605955578
    - type: f1_weighted
      value: 74.03440605955579
    - type: main_score
      value: 74.78246753246752
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB BiorxivClusteringP2P (default)
      revision: 65b79d1d13f80053f67aca9498d9402c2d9f1f40
      split: test
      type: mteb/biorxiv-clustering-p2p
    metrics:
    - type: main_score
      value: 31.887047801252244
    - type: v_measure
      value: 31.887047801252244
    - type: v_measure_std
      value: 0.5753932069603948
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB BiorxivClusteringS2S (default)
      revision: 258694dd0231531bc1fd9de6ceb52a0853c6d908
      split: test
      type: mteb/biorxiv-clustering-s2s
    metrics:
    - type: main_score
      value: 23.44412433231505
    - type: v_measure
      value: 23.44412433231505
    - type: v_measure_std
      value: 0.8476197193344371
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB CQADupstackAndroidRetrieval (default)
      revision: f46a197baaae43b4f621051089b82a364682dfeb
      split: test
      type: mteb/cqadupstack-android
    metrics:
    - type: main_score
      value: 33.725
    - type: map_at_1
      value: 21.258
    - type: map_at_10
      value: 28.599000000000004
    - type: map_at_100
      value: 29.842999999999996
    - type: map_at_1000
      value: 30.005
    - type: map_at_20
      value: 29.229
    - type: map_at_3
      value: 26.094
    - type: map_at_5
      value: 27.389000000000003
    - type: mrr_at_1
      value: 26.46638054363376
    - type: mrr_at_10
      value: 34.161387015464264
    - type: mrr_at_100
      value: 35.01880684350975
    - type: mrr_at_1000
      value: 35.097674914914485
    - type: mrr_at_20
      value: 34.60887567127352
    - type: mrr_at_3
      value: 31.75965665236053
    - type: mrr_at_5
      value: 33.0615164520744
    - type: nauc_map_at_1000_diff1
      value: 46.76618191845667
    - type: nauc_map_at_1000_max
      value: 28.936982238257396
    - type: nauc_map_at_1000_std
      value: 1.5487908974992421
    - type: nauc_map_at_100_diff1
      value: 46.79609783493257
    - type: nauc_map_at_100_max
      value: 28.91692780284247
    - type: nauc_map_at_100_std
      value: 1.502122407253288
    - type: nauc_map_at_10_diff1
      value: 46.888703900446004
    - type: nauc_map_at_10_max
      value: 28.430902905408693
    - type: nauc_map_at_10_std
      value: 1.2744648249804116
    - type: nauc_map_at_1_diff1
      value: 52.14341894573097
    - type: nauc_map_at_1_max
      value: 29.875439659067453
    - type: nauc_map_at_1_std
      value: 0.6705337537921776
    - type: nauc_map_at_20_diff1
      value: 46.673962191479795
    - type: nauc_map_at_20_max
      value: 28.6487049197731
    - type: nauc_map_at_20_std
      value: 1.1964262509403831
    - type: nauc_map_at_3_diff1
      value: 48.07832610616913
    - type: nauc_map_at_3_max
      value: 28.603262558740784
    - type: nauc_map_at_3_std
      value: 0.9437647740681423
    - type: nauc_map_at_5_diff1
      value: 47.6936940796931
    - type: nauc_map_at_5_max
      value: 28.652291541508053
    - type: nauc_map_at_5_std
      value: 0.9717878478952752
    - type: nauc_mrr_at_1000_diff1
      value: 45.33122685593024
    - type: nauc_mrr_at_1000_max
      value: 30.204338465284046
    - type: nauc_mrr_at_1000_std
      value: 2.687826356034323
    - type: nauc_mrr_at_100_diff1
      value: 45.30601560173918
    - type: nauc_mrr_at_100_max
      value: 30.18471672521032
    - type: nauc_mrr_at_100_std
      value: 2.6740730209438905
    - type: nauc_mrr_at_10_diff1
      value: 45.41931593964348
    - type: nauc_mrr_at_10_max
      value: 30.227605387613377
    - type: nauc_mrr_at_10_std
      value: 2.5467078314775105
    - type: nauc_mrr_at_1_diff1
      value: 52.578617006402695
    - type: nauc_mrr_at_1_max
      value: 31.533124113425608
    - type: nauc_mrr_at_1_std
      value: 2.142001651137791
    - type: nauc_mrr_at_20_diff1
      value: 45.1567569739636
    - type: nauc_mrr_at_20_max
      value: 30.068202057592075
    - type: nauc_mrr_at_20_std
      value: 2.498778251276313
    - type: nauc_mrr_at_3_diff1
      value: 46.53010950514913
    - type: nauc_mrr_at_3_max
      value: 30.55396071943546
    - type: nauc_mrr_at_3_std
      value: 2.5301194724381775
    - type: nauc_mrr_at_5_diff1
      value: 46.05508257170174
    - type: nauc_mrr_at_5_max
      value: 30.778384564258776
    - type: nauc_mrr_at_5_std
      value: 2.558309698641406
    - type: nauc_ndcg_at_1000_diff1
      value: 43.753900702619724
    - type: nauc_ndcg_at_1000_max
      value: 29.633265380008684
    - type: nauc_ndcg_at_1000_std
      value: 4.486049141568419
    - type: nauc_ndcg_at_100_diff1
      value: 43.62494408120729
    - type: nauc_ndcg_at_100_max
      value: 29.21612586326204
    - type: nauc_ndcg_at_100_std
      value: 3.8426617907301974
    - type: nauc_ndcg_at_10_diff1
      value: 43.55664235851717
    - type: nauc_ndcg_at_10_max
      value: 27.907959174030626
    - type: nauc_ndcg_at_10_std
      value: 1.9864038329637217
    - type: nauc_ndcg_at_1_diff1
      value: 52.578617006402695
    - type: nauc_ndcg_at_1_max
      value: 31.533124113425608
    - type: nauc_ndcg_at_1_std
      value: 2.142001651137791
    - type: nauc_ndcg_at_20_diff1
      value: 42.83241987465397
    - type: nauc_ndcg_at_20_max
      value: 27.88256330396997
    - type: nauc_ndcg_at_20_std
      value: 1.7703781723570542
    - type: nauc_ndcg_at_3_diff1
      value: 45.4190097736324
    - type: nauc_ndcg_at_3_max
      value: 28.560425888173796
    - type: nauc_ndcg_at_3_std
      value: 1.854268064126404
    - type: nauc_ndcg_at_5_diff1
      value: 44.98606135986684
    - type: nauc_ndcg_at_5_max
      value: 28.566365021440337
    - type: nauc_ndcg_at_5_std
      value: 1.6742805472789761
    - type: nauc_precision_at_1000_diff1
      value: -5.4841077392281035
    - type: nauc_precision_at_1000_max
      value: -0.34081891189369173
    - type: nauc_precision_at_1000_std
      value: 2.1036736091111585
    - type: nauc_precision_at_100_diff1
      value: 7.441486720589044
    - type: nauc_precision_at_100_max
      value: 13.334970101878652
    - type: nauc_precision_at_100_std
      value: 6.9306352695965066
    - type: nauc_precision_at_10_diff1
      value: 24.853282788022366
    - type: nauc_precision_at_10_max
      value: 23.45632252018925
    - type: nauc_precision_at_10_std
      value: 3.7432075056706267
    - type: nauc_precision_at_1_diff1
      value: 52.578617006402695
    - type: nauc_precision_at_1_max
      value: 31.533124113425608
    - type: nauc_precision_at_1_std
      value: 2.142001651137791
    - type: nauc_precision_at_20_diff1
      value: 16.874948719815144
    - type: nauc_precision_at_20_max
      value: 20.186814805341797
    - type: nauc_precision_at_20_std
      value: 1.996681050839198
    - type: nauc_precision_at_3_diff1
      value: 38.02476874042044
    - type: nauc_precision_at_3_max
      value: 27.923642221335314
    - type: nauc_precision_at_3_std
      value: 1.841951122412098
    - type: nauc_precision_at_5_diff1
      value: 34.257705852347975
    - type: nauc_precision_at_5_max
      value: 26.51537237704359
    - type: nauc_precision_at_5_std
      value: 2.1637726175663627
    - type: nauc_recall_at_1000_diff1
      value: 28.158519011741966
    - type: nauc_recall_at_1000_max
      value: 33.26807338931001
    - type: nauc_recall_at_1000_std
      value: 38.0648935642973
    - type: nauc_recall_at_100_diff1
      value: 30.0156168828398
    - type: nauc_recall_at_100_max
      value: 26.250024559731
    - type: nauc_recall_at_100_std
      value: 14.034527192600873
    - type: nauc_recall_at_10_diff1
      value: 33.362811050628714
    - type: nauc_recall_at_10_max
      value: 22.08581852634732
    - type: nauc_recall_at_10_std
      value: 3.287335910498459
    - type: nauc_recall_at_1_diff1
      value: 52.14341894573097
    - type: nauc_recall_at_1_max
      value: 29.875439659067453
    - type: nauc_recall_at_1_std
      value: 0.6705337537921776
    - type: nauc_recall_at_20_diff1
      value: 29.9377683089396
    - type: nauc_recall_at_20_max
      value: 21.501166512666366
    - type: nauc_recall_at_20_std
      value: 2.5674343420113637
    - type: nauc_recall_at_3_diff1
      value: 40.61950305751733
    - type: nauc_recall_at_3_max
      value: 24.729983168436682
    - type: nauc_recall_at_3_std
      value: 1.7296166060546279
    - type: nauc_recall_at_5_diff1
      value: 38.552821903528056
    - type: nauc_recall_at_5_max
      value: 24.723317875854455
    - type: nauc_recall_at_5_std
      value: 1.919467574179939
    - type: ndcg_at_1
      value: 26.466
    - type: ndcg_at_10
      value: 33.725
    - type: ndcg_at_100
      value: 39.173
    - type: ndcg_at_1000
      value: 42.232
    - type: ndcg_at_20
      value: 35.567
    - type: ndcg_at_3
      value: 29.809
    - type: ndcg_at_5
      value: 31.34
    - type: precision_at_1
      value: 26.466
    - type: precision_at_10
      value: 6.465999999999999
    - type: precision_at_100
      value: 1.157
    - type: precision_at_1000
      value: 0.17500000000000002
    - type: precision_at_20
      value: 3.9059999999999997
    - type: precision_at_3
      value: 14.449000000000002
    - type: precision_at_5
      value: 10.358
    - type: recall_at_1
      value: 21.258
    - type: recall_at_10
      value: 43.312
    - type: recall_at_100
      value: 67.238
    - type: recall_at_1000
      value: 87.595
    - type: recall_at_20
      value: 50.041999999999994
    - type: recall_at_3
      value: 31.159
    - type: recall_at_5
      value: 35.879
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackEnglishRetrieval (default)
      revision: ad9991cb51e31e31e430383c75ffb2885547b5f0
      split: test
      type: mteb/cqadupstack-english
    metrics:
    - type: main_score
      value: 27.150000000000002
    - type: map_at_1
      value: 17.43
    - type: map_at_10
      value: 23.204
    - type: map_at_100
      value: 24.145
    - type: map_at_1000
      value: 24.265
    - type: map_at_20
      value: 23.686
    - type: map_at_3
      value: 21.189
    - type: map_at_5
      value: 22.255
    - type: mrr_at_1
      value: 21.719745222929937
    - type: mrr_at_10
      value: 27.507557375391766
    - type: mrr_at_100
      value: 28.30013304854397
    - type: mrr_at_1000
      value: 28.368740901376388
    - type: mrr_at_20
      value: 27.951898243025873
    - type: mrr_at_3
      value: 25.594479830148607
    - type: mrr_at_5
      value: 26.63588110403395
    - type: nauc_map_at_1000_diff1
      value: 40.11409523933898
    - type: nauc_map_at_1000_max
      value: 16.374923278675748
    - type: nauc_map_at_1000_std
      value: -2.4133717957936653
    - type: nauc_map_at_100_diff1
      value: 40.162705532329554
    - type: nauc_map_at_100_max
      value: 16.34063524581922
    - type: nauc_map_at_100_std
      value: -2.493868733681066
    - type: nauc_map_at_10_diff1
      value: 40.48041468478108
    - type: nauc_map_at_10_max
      value: 16.264842404610157
    - type: nauc_map_at_10_std
      value: -3.2700352899130314
    - type: nauc_map_at_1_diff1
      value: 46.96738086855307
    - type: nauc_map_at_1_max
      value: 17.12380109224913
    - type: nauc_map_at_1_std
      value: -4.311116279351113
    - type: nauc_map_at_20_diff1
      value: 40.27243337041536
    - type: nauc_map_at_20_max
      value: 16.212635684878002
    - type: nauc_map_at_20_std
      value: -2.8818599939377325
    - type: nauc_map_at_3_diff1
      value: 41.701659705049074
    - type: nauc_map_at_3_max
      value: 16.359651564201815
    - type: nauc_map_at_3_std
      value: -4.294683564342943
    - type: nauc_map_at_5_diff1
      value: 41.028355766533195
    - type: nauc_map_at_5_max
      value: 16.27940855548611
    - type: nauc_map_at_5_std
      value: -3.8828196073115726
    - type: nauc_mrr_at_1000_diff1
      value: 37.35241750993703
    - type: nauc_mrr_at_1000_max
      value: 16.789493628813503
    - type: nauc_mrr_at_1000_std
      value: -0.7026807998359369
    - type: nauc_mrr_at_100_diff1
      value: 37.37319265054083
    - type: nauc_mrr_at_100_max
      value: 16.785792904909457
    - type: nauc_mrr_at_100_std
      value: -0.7141300613570387
    - type: nauc_mrr_at_10_diff1
      value: 37.44022864370511
    - type: nauc_mrr_at_10_max
      value: 16.822238004660257
    - type: nauc_mrr_at_10_std
      value: -1.064281278734109
    - type: nauc_mrr_at_1_diff1
      value: 42.576217828618965
    - type: nauc_mrr_at_1_max
      value: 20.139157671233285
    - type: nauc_mrr_at_1_std
      value: -1.1737759527663842
    - type: nauc_mrr_at_20_diff1
      value: 37.33937055395397
    - type: nauc_mrr_at_20_max
      value: 16.776195886601606
    - type: nauc_mrr_at_20_std
      value: -0.7803878999084822
    - type: nauc_mrr_at_3_diff1
      value: 38.25326211165273
    - type: nauc_mrr_at_3_max
      value: 17.15336690620075
    - type: nauc_mrr_at_3_std
      value: -1.2801443058247721
    - type: nauc_mrr_at_5_diff1
      value: 38.06596720050976
    - type: nauc_mrr_at_5_max
      value: 17.09392231006509
    - type: nauc_mrr_at_5_std
      value: -1.3623433653714556
    - type: nauc_ndcg_at_1000_diff1
      value: 36.435133711331915
    - type: nauc_ndcg_at_1000_max
      value: 15.586688731412274
    - type: nauc_ndcg_at_1000_std
      value: 0.7906560302800849
    - type: nauc_ndcg_at_100_diff1
      value: 36.890721464008045
    - type: nauc_ndcg_at_100_max
      value: 15.1813884843887
    - type: nauc_ndcg_at_100_std
      value: 0.09040811909091676
    - type: nauc_ndcg_at_10_diff1
      value: 37.229997064998585
    - type: nauc_ndcg_at_10_max
      value: 15.451385774929092
    - type: nauc_ndcg_at_10_std
      value: -1.8785020804451147
    - type: nauc_ndcg_at_1_diff1
      value: 42.576217828618965
    - type: nauc_ndcg_at_1_max
      value: 20.139157671233285
    - type: nauc_ndcg_at_1_std
      value: -1.1737759527663842
    - type: nauc_ndcg_at_20_diff1
      value: 36.838912228068594
    - type: nauc_ndcg_at_20_max
      value: 14.995726844190102
    - type: nauc_ndcg_at_20_std
      value: -1.0539261339339805
    - type: nauc_ndcg_at_3_diff1
      value: 38.82534211536086
    - type: nauc_ndcg_at_3_max
      value: 16.220832428536855
    - type: nauc_ndcg_at_3_std
      value: -2.813569063131948
    - type: nauc_ndcg_at_5_diff1
      value: 38.33996404125124
    - type: nauc_ndcg_at_5_max
      value: 15.8799422475145
    - type: nauc_ndcg_at_5_std
      value: -2.749146560430897
    - type: nauc_precision_at_1000_diff1
      value: -4.731735053924757
    - type: nauc_precision_at_1000_max
      value: 7.435193268192747
    - type: nauc_precision_at_1000_std
      value: 19.64454136253714
    - type: nauc_precision_at_100_diff1
      value: 6.077831904313305
    - type: nauc_precision_at_100_max
      value: 11.106929057800805
    - type: nauc_precision_at_100_std
      value: 18.011167821410282
    - type: nauc_precision_at_10_diff1
      value: 20.063107688804948
    - type: nauc_precision_at_10_max
      value: 14.59451412625624
    - type: nauc_precision_at_10_std
      value: 6.453563891426743
    - type: nauc_precision_at_1_diff1
      value: 42.576217828618965
    - type: nauc_precision_at_1_max
      value: 20.139157671233285
    - type: nauc_precision_at_1_std
      value: -1.1737759527663842
    - type: nauc_precision_at_20_diff1
      value: 13.959222956715548
    - type: nauc_precision_at_20_max
      value: 12.477724590330006
    - type: nauc_precision_at_20_std
      value: 11.336423270774759
    - type: nauc_precision_at_3_diff1
      value: 29.664870607877365
    - type: nauc_precision_at_3_max
      value: 16.707672459588675
    - type: nauc_precision_at_3_std
      value: 1.2390528951961652
    - type: nauc_precision_at_5_diff1
      value: 26.04656621005802
    - type: nauc_precision_at_5_max
      value: 16.277009527866586
    - type: nauc_precision_at_5_std
      value: 3.1802968656941237
    - type: nauc_recall_at_1000_diff1
      value: 22.61803986653323
    - type: nauc_recall_at_1000_max
      value: 10.862168120090677
    - type: nauc_recall_at_1000_std
      value: 11.350094310700353
    - type: nauc_recall_at_100_diff1
      value: 27.499885478435775
    - type: nauc_recall_at_100_max
      value: 9.642182763633409
    - type: nauc_recall_at_100_std
      value: 5.863748205800293
    - type: nauc_recall_at_10_diff1
      value: 30.529857718433185
    - type: nauc_recall_at_10_max
      value: 11.752541699752392
    - type: nauc_recall_at_10_std
      value: -1.280154699097079
    - type: nauc_recall_at_1_diff1
      value: 46.96738086855307
    - type: nauc_recall_at_1_max
      value: 17.12380109224913
    - type: nauc_recall_at_1_std
      value: -4.311116279351113
    - type: nauc_recall_at_20_diff1
      value: 28.358327134514628
    - type: nauc_recall_at_20_max
      value: 9.90049047345775
    - type: nauc_recall_at_20_std
      value: 1.3182686690583876
    - type: nauc_recall_at_3_diff1
      value: 36.19909796459414
    - type: nauc_recall_at_3_max
      value: 13.526137335233242
    - type: nauc_recall_at_3_std
      value: -4.332223904768279
    - type: nauc_recall_at_5_diff1
      value: 34.40402552836485
    - type: nauc_recall_at_5_max
      value: 12.723314342847472
    - type: nauc_recall_at_5_std
      value: -3.782494828035306
    - type: ndcg_at_1
      value: 21.72
    - type: ndcg_at_10
      value: 27.150000000000002
    - type: ndcg_at_100
      value: 31.439
    - type: ndcg_at_1000
      value: 34.277
    - type: ndcg_at_20
      value: 28.663
    - type: ndcg_at_3
      value: 23.726
    - type: ndcg_at_5
      value: 25.189
    - type: precision_at_1
      value: 21.72
    - type: precision_at_10
      value: 5.0
    - type: precision_at_100
      value: 0.907
    - type: precision_at_1000
      value: 0.14100000000000001
    - type: precision_at_20
      value: 3.057
    - type: precision_at_3
      value: 11.04
    - type: precision_at_5
      value: 7.9750000000000005
    - type: recall_at_1
      value: 17.43
    - type: recall_at_10
      value: 34.688
    - type: recall_at_100
      value: 53.301
    - type: recall_at_1000
      value: 72.772
    - type: recall_at_20
      value: 40.198
    - type: recall_at_3
      value: 24.982
    - type: recall_at_5
      value: 28.786
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackGamingRetrieval (default)
      revision: 4885aa143210c98657558c04aaf3dc47cfb54340
      split: test
      type: mteb/cqadupstack-gaming
    metrics:
    - type: main_score
      value: 41.524
    - type: map_at_1
      value: 27.182000000000002
    - type: map_at_10
      value: 36.466
    - type: map_at_100
      value: 37.509
    - type: map_at_1000
      value: 37.601
    - type: map_at_20
      value: 37.013
    - type: map_at_3
      value: 33.668
    - type: map_at_5
      value: 35.397
    - type: mrr_at_1
      value: 31.41065830721003
    - type: mrr_at_10
      value: 39.7386674628054
    - type: mrr_at_100
      value: 40.566291942215805
    - type: mrr_at_1000
      value: 40.62074545423136
    - type: mrr_at_20
      value: 40.174934295344926
    - type: mrr_at_3
      value: 37.324973876697996
    - type: mrr_at_5
      value: 38.826541274817146
    - type: nauc_map_at_1000_diff1
      value: 43.6362711634798
    - type: nauc_map_at_1000_max
      value: 26.519736629780972
    - type: nauc_map_at_1000_std
      value: -9.384141101744808
    - type: nauc_map_at_100_diff1
      value: 43.61961122855754
    - type: nauc_map_at_100_max
      value: 26.487007810288453
    - type: nauc_map_at_100_std
      value: -9.420758617427817
    - type: nauc_map_at_10_diff1
      value: 43.7524147747428
    - type: nauc_map_at_10_max
      value: 26.382789751233844
    - type: nauc_map_at_10_std
      value: -9.769072712489033
    - type: nauc_map_at_1_diff1
      value: 47.04811157014266
    - type: nauc_map_at_1_max
      value: 21.54129949413187
    - type: nauc_map_at_1_std
      value: -12.019452027355847
    - type: nauc_map_at_20_diff1
      value: 43.672216947782175
    - type: nauc_map_at_20_max
      value: 26.485649204915397
    - type: nauc_map_at_20_std
      value: -9.524911676842086
    - type: nauc_map_at_3_diff1
      value: 44.348004771454576
    - type: nauc_map_at_3_max
      value: 24.826912771861814
    - type: nauc_map_at_3_std
      value: -10.965059117490936
    - type: nauc_map_at_5_diff1
      value: 43.99455131879081
    - type: nauc_map_at_5_max
      value: 25.971868217929885
    - type: nauc_map_at_5_std
      value: -10.343047286710718
    - type: nauc_mrr_at_1000_diff1
      value: 43.665788051551665
    - type: nauc_mrr_at_1000_max
      value: 28.571240161784754
    - type: nauc_mrr_at_1000_std
      value: -7.4113510667337525
    - type: nauc_mrr_at_100_diff1
      value: 43.64279734673117
    - type: nauc_mrr_at_100_max
      value: 28.567634128513596
    - type: nauc_mrr_at_100_std
      value: -7.40852157646149
    - type: nauc_mrr_at_10_diff1
      value: 43.782023975278136
    - type: nauc_mrr_at_10_max
      value: 28.790688045761286
    - type: nauc_mrr_at_10_std
      value: -7.440405024660997
    - type: nauc_mrr_at_1_diff1
      value: 47.695888805627476
    - type: nauc_mrr_at_1_max
      value: 25.82725382005077
    - type: nauc_mrr_at_1_std
      value: -9.926228290630222
    - type: nauc_mrr_at_20_diff1
      value: 43.70864148086972
    - type: nauc_mrr_at_20_max
      value: 28.624617221296354
    - type: nauc_mrr_at_20_std
      value: -7.4340361462335665
    - type: nauc_mrr_at_3_diff1
      value: 44.074855443552025
    - type: nauc_mrr_at_3_max
      value: 27.910450462219337
    - type: nauc_mrr_at_3_std
      value: -8.482348718508304
    - type: nauc_mrr_at_5_diff1
      value: 43.982929810180146
    - type: nauc_mrr_at_5_max
      value: 28.744111522007042
    - type: nauc_mrr_at_5_std
      value: -7.787954236435398
    - type: nauc_ndcg_at_1000_diff1
      value: 42.100416444667374
    - type: nauc_ndcg_at_1000_max
      value: 28.778897448421066
    - type: nauc_ndcg_at_1000_std
      value: -5.940439957350017
    - type: nauc_ndcg_at_100_diff1
      value: 41.67916983287973
    - type: nauc_ndcg_at_100_max
      value: 28.391295000413553
    - type: nauc_ndcg_at_100_std
      value: -6.3451034687120735
    - type: nauc_ndcg_at_10_diff1
      value: 42.43970626182128
    - type: nauc_ndcg_at_10_max
      value: 28.6158195146179
    - type: nauc_ndcg_at_10_std
      value: -7.6406087703592505
    - type: nauc_ndcg_at_1_diff1
      value: 47.695888805627476
    - type: nauc_ndcg_at_1_max
      value: 25.82725382005077
    - type: nauc_ndcg_at_1_std
      value: -9.926228290630222
    - type: nauc_ndcg_at_20_diff1
      value: 42.03681959175668
    - type: nauc_ndcg_at_20_max
      value: 28.453546652524885
    - type: nauc_ndcg_at_20_std
      value: -7.152848103502503
    - type: nauc_ndcg_at_3_diff1
      value: 43.37129896398724
    - type: nauc_ndcg_at_3_max
      value: 26.478953479700902
    - type: nauc_ndcg_at_3_std
      value: -9.679309883071229
    - type: nauc_ndcg_at_5_diff1
      value: 43.0062039655728
    - type: nauc_ndcg_at_5_max
      value: 28.064633302237336
    - type: nauc_ndcg_at_5_std
      value: -8.7910164137182
    - type: nauc_precision_at_1000_diff1
      value: -0.5507173446795858
    - type: nauc_precision_at_1000_max
      value: 22.126966541299343
    - type: nauc_precision_at_1000_std
      value: 20.148121474343323
    - type: nauc_precision_at_100_diff1
      value: 10.44995531655567
    - type: nauc_precision_at_100_max
      value: 26.56665886767694
    - type: nauc_precision_at_100_std
      value: 12.195696500074583
    - type: nauc_precision_at_10_diff1
      value: 26.158452845146336
    - type: nauc_precision_at_10_max
      value: 32.01459975128394
    - type: nauc_precision_at_10_std
      value: 1.974561798960782
    - type: nauc_precision_at_1_diff1
      value: 47.695888805627476
    - type: nauc_precision_at_1_max
      value: 25.82725382005077
    - type: nauc_precision_at_1_std
      value: -9.926228290630222
    - type: nauc_precision_at_20_diff1
      value: 22.032211497868598
    - type: nauc_precision_at_20_max
      value: 31.233368398218488
    - type: nauc_precision_at_20_std
      value: 6.013804577433131
    - type: nauc_precision_at_3_diff1
      value: 35.54675021433468
    - type: nauc_precision_at_3_max
      value: 30.674121449268544
    - type: nauc_precision_at_3_std
      value: -5.765186040985941
    - type: nauc_precision_at_5_diff1
      value: 31.458592549241732
    - type: nauc_precision_at_5_max
      value: 32.97315493502319
    - type: nauc_precision_at_5_std
      value: -2.541487737695983
    - type: nauc_recall_at_1000_diff1
      value: 26.304660389426214
    - type: nauc_recall_at_1000_max
      value: 40.401428843868544
    - type: nauc_recall_at_1000_std
      value: 32.529641304158815
    - type: nauc_recall_at_100_diff1
      value: 29.344646248064482
    - type: nauc_recall_at_100_max
      value: 30.235178076671676
    - type: nauc_recall_at_100_std
      value: 7.52291487382479
    - type: nauc_recall_at_10_diff1
      value: 36.611201944418376
    - type: nauc_recall_at_10_max
      value: 31.24170076999929
    - type: nauc_recall_at_10_std
      value: -2.9884690234741784
    - type: nauc_recall_at_1_diff1
      value: 47.04811157014266
    - type: nauc_recall_at_1_max
      value: 21.54129949413187
    - type: nauc_recall_at_1_std
      value: -12.019452027355847
    - type: nauc_recall_at_20_diff1
      value: 34.63747670575681
    - type: nauc_recall_at_20_max
      value: 30.43743109165214
    - type: nauc_recall_at_20_std
      value: -1.227195536805767
    - type: nauc_recall_at_3_diff1
      value: 39.91051874486281
    - type: nauc_recall_at_3_max
      value: 26.560414903761114
    - type: nauc_recall_at_3_std
      value: -8.6755601946905
    - type: nauc_recall_at_5_diff1
      value: 38.75064222937078
    - type: nauc_recall_at_5_max
      value: 29.845489039156003
    - type: nauc_recall_at_5_std
      value: -6.453948528328865
    - type: ndcg_at_1
      value: 31.411
    - type: ndcg_at_10
      value: 41.524
    - type: ndcg_at_100
      value: 46.504
    - type: ndcg_at_1000
      value: 48.597
    - type: ndcg_at_20
      value: 43.256
    - type: ndcg_at_3
      value: 36.579
    - type: ndcg_at_5
      value: 39.278
    - type: precision_at_1
      value: 31.411
    - type: precision_at_10
      value: 6.8709999999999996
    - type: precision_at_100
      value: 1.023
    - type: precision_at_1000
      value: 0.128
    - type: precision_at_20
      value: 3.9149999999999996
    - type: precision_at_3
      value: 16.384999999999998
    - type: precision_at_5
      value: 11.687
    - type: recall_at_1
      value: 27.182000000000002
    - type: recall_at_10
      value: 53.385000000000005
    - type: recall_at_100
      value: 76.191
    - type: recall_at_1000
      value: 91.365
    - type: recall_at_20
      value: 59.953
    - type: recall_at_3
      value: 40.388000000000005
    - type: recall_at_5
      value: 46.885
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackGisRetrieval (default)
      revision: 5003b3064772da1887988e05400cf3806fe491f2
      split: test
      type: mteb/cqadupstack-gis
    metrics:
    - type: main_score
      value: 20.852999999999998
    - type: map_at_1
      value: 12.659
    - type: map_at_10
      value: 17.837
    - type: map_at_100
      value: 18.619
    - type: map_at_1000
      value: 18.742
    - type: map_at_20
      value: 18.257
    - type: map_at_3
      value: 16.283
    - type: map_at_5
      value: 17.335
    - type: mrr_at_1
      value: 13.785310734463277
    - type: mrr_at_10
      value: 19.169491525423734
    - type: mrr_at_100
      value: 19.95101655525465
    - type: mrr_at_1000
      value: 20.056219981020202
    - type: mrr_at_20
      value: 19.592521112159574
    - type: mrr_at_3
      value: 17.551789077212806
    - type: mrr_at_5
      value: 18.687382297551792
    - type: nauc_map_at_1000_diff1
      value: 32.97526567584334
    - type: nauc_map_at_1000_max
      value: 17.827080363908074
    - type: nauc_map_at_1000_std
      value: -7.917256868134976
    - type: nauc_map_at_100_diff1
      value: 33.00071714912684
    - type: nauc_map_at_100_max
      value: 17.77244247160442
    - type: nauc_map_at_100_std
      value: -7.954025554315228
    - type: nauc_map_at_10_diff1
      value: 33.540719726361566
    - type: nauc_map_at_10_max
      value: 17.985491695772446
    - type: nauc_map_at_10_std
      value: -7.856378803376327
    - type: nauc_map_at_1_diff1
      value: 43.67494937362112
    - type: nauc_map_at_1_max
      value: 21.124340797673945
    - type: nauc_map_at_1_std
      value: -12.197996046930768
    - type: nauc_map_at_20_diff1
      value: 33.223910724903206
    - type: nauc_map_at_20_max
      value: 17.886404791497466
    - type: nauc_map_at_20_std
      value: -8.16041395141026
    - type: nauc_map_at_3_diff1
      value: 35.72043899824334
    - type: nauc_map_at_3_max
      value: 18.432956304616784
    - type: nauc_map_at_3_std
      value: -9.080010089944173
    - type: nauc_map_at_5_diff1
      value: 33.971592181962734
    - type: nauc_map_at_5_max
      value: 18.424185588555975
    - type: nauc_map_at_5_std
      value: -7.577770606753287
    - type: nauc_mrr_at_1000_diff1
      value: 31.433579780624594
    - type: nauc_mrr_at_1000_max
      value: 19.322453443072256
    - type: nauc_mrr_at_1000_std
      value: -5.854685619590339
    - type: nauc_mrr_at_100_diff1
      value: 31.445404071603033
    - type: nauc_mrr_at_100_max
      value: 19.304722387301677
    - type: nauc_mrr_at_100_std
      value: -5.8736348679902175
    - type: nauc_mrr_at_10_diff1
      value: 31.83707006941321
    - type: nauc_mrr_at_10_max
      value: 19.34188797300804
    - type: nauc_mrr_at_10_std
      value: -5.745261013451921
    - type: nauc_mrr_at_1_diff1
      value: 40.93685635066508
    - type: nauc_mrr_at_1_max
      value: 23.439679209668945
    - type: nauc_mrr_at_1_std
      value: -9.177572150758774
    - type: nauc_mrr_at_20_diff1
      value: 31.592328454500933
    - type: nauc_mrr_at_20_max
      value: 19.36391895653557
    - type: nauc_mrr_at_20_std
      value: -6.040902065763658
    - type: nauc_mrr_at_3_diff1
      value: 33.84195578174624
    - type: nauc_mrr_at_3_max
      value: 19.72761095405792
    - type: nauc_mrr_at_3_std
      value: -6.874819526162579
    - type: nauc_mrr_at_5_diff1
      value: 32.236409494283755
    - type: nauc_mrr_at_5_max
      value: 19.697954589210916
    - type: nauc_mrr_at_5_std
      value: -5.4516874898461625
    - type: nauc_ndcg_at_1000_diff1
      value: 27.505013952943784
    - type: nauc_ndcg_at_1000_max
      value: 16.80970010826237
    - type: nauc_ndcg_at_1000_std
      value: -5.331859971693516
    - type: nauc_ndcg_at_100_diff1
      value: 27.503387634967602
    - type: nauc_ndcg_at_100_max
      value: 15.916003406666354
    - type: nauc_ndcg_at_100_std
      value: -6.1221427231558145
    - type: nauc_ndcg_at_10_diff1
      value: 29.664304651922325
    - type: nauc_ndcg_at_10_max
      value: 17.007347849411076
    - type: nauc_ndcg_at_10_std
      value: -6.266816577956439
    - type: nauc_ndcg_at_1_diff1
      value: 40.93685635066508
    - type: nauc_ndcg_at_1_max
      value: 23.439679209668945
    - type: nauc_ndcg_at_1_std
      value: -9.177572150758774
    - type: nauc_ndcg_at_20_diff1
      value: 28.647331375815643
    - type: nauc_ndcg_at_20_max
      value: 16.87787934591494
    - type: nauc_ndcg_at_20_std
      value: -7.258408352703308
    - type: nauc_ndcg_at_3_diff1
      value: 33.38934212428272
    - type: nauc_ndcg_at_3_max
      value: 17.91982977008598
    - type: nauc_ndcg_at_3_std
      value: -8.009957293234983
    - type: nauc_ndcg_at_5_diff1
      value: 30.61169550826665
    - type: nauc_ndcg_at_5_max
      value: 17.91887589124064
    - type: nauc_ndcg_at_5_std
      value: -5.585432013144523
    - type: nauc_precision_at_1000_diff1
      value: 0.4226603303036744
    - type: nauc_precision_at_1000_max
      value: 14.332893022601741
    - type: nauc_precision_at_1000_std
      value: 9.035818125389602
    - type: nauc_precision_at_100_diff1
      value: 10.538061012787567
    - type: nauc_precision_at_100_max
      value: 11.494339713810401
    - type: nauc_precision_at_100_std
      value: 1.4164358406768551
    - type: nauc_precision_at_10_diff1
      value: 19.644828725115065
    - type: nauc_precision_at_10_max
      value: 15.64894166408731
    - type: nauc_precision_at_10_std
      value: -1.081339939239084
    - type: nauc_precision_at_1_diff1
      value: 40.93685635066508
    - type: nauc_precision_at_1_max
      value: 23.439679209668945
    - type: nauc_precision_at_1_std
      value: -9.177572150758774
    - type: nauc_precision_at_20_diff1
      value: 16.383600154012903
    - type: nauc_precision_at_20_max
      value: 14.882146834330158
    - type: nauc_precision_at_20_std
      value: -3.5303919234659564
    - type: nauc_precision_at_3_diff1
      value: 26.300673270530456
    - type: nauc_precision_at_3_max
      value: 17.728543486739575
    - type: nauc_precision_at_3_std
      value: -4.398100967559122
    - type: nauc_precision_at_5_diff1
      value: 21.490299955788664
    - type: nauc_precision_at_5_max
      value: 17.908106757943347
    - type: nauc_precision_at_5_std
      value: 0.42728730989335834
    - type: nauc_recall_at_1000_diff1
      value: 7.740221752377091
    - type: nauc_recall_at_1000_max
      value: 10.555708189438292
    - type: nauc_recall_at_1000_std
      value: 5.695404463900641
    - type: nauc_recall_at_100_diff1
      value: 13.228097861724095
    - type: nauc_recall_at_100_max
      value: 8.046587601353371
    - type: nauc_recall_at_100_std
      value: -2.8441726501064792
    - type: nauc_recall_at_10_diff1
      value: 21.245766737033776
    - type: nauc_recall_at_10_max
      value: 13.355463123202746
    - type: nauc_recall_at_10_std
      value: -4.306099687448025
    - type: nauc_recall_at_1_diff1
      value: 43.67494937362112
    - type: nauc_recall_at_1_max
      value: 21.124340797673945
    - type: nauc_recall_at_1_std
      value: -12.197996046930768
    - type: nauc_recall_at_20_diff1
      value: 18.077039106796608
    - type: nauc_recall_at_20_max
      value: 13.019919308285566
    - type: nauc_recall_at_20_std
      value: -7.4206526771047
    - type: nauc_recall_at_3_diff1
      value: 29.086042342675317
    - type: nauc_recall_at_3_max
      value: 15.5421583851582
    - type: nauc_recall_at_3_std
      value: -7.123318203911006
    - type: nauc_recall_at_5_diff1
      value: 23.40979964056917
    - type: nauc_recall_at_5_max
      value: 15.430969619214412
    - type: nauc_recall_at_5_std
      value: -2.4671514179639678
    - type: ndcg_at_1
      value: 13.785
    - type: ndcg_at_10
      value: 20.852999999999998
    - type: ndcg_at_100
      value: 25.115
    - type: ndcg_at_1000
      value: 28.534
    - type: ndcg_at_20
      value: 22.367
    - type: ndcg_at_3
      value: 17.775
    - type: ndcg_at_5
      value: 19.674
    - type: precision_at_1
      value: 13.785
    - type: precision_at_10
      value: 3.266
    - type: precision_at_100
      value: 0.573
    - type: precision_at_1000
      value: 0.091
    - type: precision_at_20
      value: 1.994
    - type: precision_at_3
      value: 7.721
    - type: precision_at_5
      value: 5.695
    - type: recall_at_1
      value: 12.659
    - type: recall_at_10
      value: 28.828
    - type: recall_at_100
      value: 49.171
    - type: recall_at_1000
      value: 75.594
    - type: recall_at_20
      value: 34.544999999999995
    - type: recall_at_3
      value: 20.754
    - type: recall_at_5
      value: 25.252999999999997
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackMathematicaRetrieval (default)
      revision: 90fceea13679c63fe563ded68f3b6f06e50061de
      split: test
      type: mteb/cqadupstack-mathematica
    metrics:
    - type: main_score
      value: 13.624
    - type: map_at_1
      value: 7.0809999999999995
    - type: map_at_10
      value: 10.864
    - type: map_at_100
      value: 11.705
    - type: map_at_1000
      value: 11.838
    - type: map_at_20
      value: 11.251999999999999
    - type: map_at_3
      value: 9.383
    - type: map_at_5
      value: 10.235
    - type: mrr_at_1
      value: 8.582089552238806
    - type: mrr_at_10
      value: 12.98472913211719
    - type: mrr_at_100
      value: 13.850379213620226
    - type: mrr_at_1000
      value: 13.952864877234141
    - type: mrr_at_20
      value: 13.413476285325437
    - type: mrr_at_3
      value: 11.38059701492537
    - type: mrr_at_5
      value: 12.251243781094525
    - type: nauc_map_at_1000_diff1
      value: 13.869076012836793
    - type: nauc_map_at_1000_max
      value: 9.793728325817751
    - type: nauc_map_at_1000_std
      value: 4.900696283931284
    - type: nauc_map_at_100_diff1
      value: 13.860320769496532
    - type: nauc_map_at_100_max
      value: 9.686534228015688
    - type: nauc_map_at_100_std
      value: 4.810028020121735
    - type: nauc_map_at_10_diff1
      value: 13.143868194306984
    - type: nauc_map_at_10_max
      value: 9.373464306802715
    - type: nauc_map_at_10_std
      value: 4.079400193707048
    - type: nauc_map_at_1_diff1
      value: 24.022346561431156
    - type: nauc_map_at_1_max
      value: 7.841750008744963
    - type: nauc_map_at_1_std
      value: 3.3784810578492785
    - type: nauc_map_at_20_diff1
      value: 13.635585210806106
    - type: nauc_map_at_20_max
      value: 9.752805074800094
    - type: nauc_map_at_20_std
      value: 4.654472956329851
    - type: nauc_map_at_3_diff1
      value: 16.85106726903103
    - type: nauc_map_at_3_max
      value: 8.426565488274038
    - type: nauc_map_at_3_std
      value: 5.022674813566249
    - type: nauc_map_at_5_diff1
      value: 14.134393994697025
    - type: nauc_map_at_5_max
      value: 9.49019400022355
    - type: nauc_map_at_5_std
      value: 4.293050389455758
    - type: nauc_mrr_at_1000_diff1
      value: 15.430729302655086
    - type: nauc_mrr_at_1000_max
      value: 11.130235636889111
    - type: nauc_mrr_at_1000_std
      value: 4.877791549279745
    - type: nauc_mrr_at_100_diff1
      value: 15.410324011690738
    - type: nauc_mrr_at_100_max
      value: 11.068477306407296
    - type: nauc_mrr_at_100_std
      value: 4.843658916752368
    - type: nauc_mrr_at_10_diff1
      value: 15.030861163034931
    - type: nauc_mrr_at_10_max
      value: 10.949618861931153
    - type: nauc_mrr_at_10_std
      value: 4.688892607587696
    - type: nauc_mrr_at_1_diff1
      value: 24.902916052765633
    - type: nauc_mrr_at_1_max
      value: 9.457290628689096
    - type: nauc_mrr_at_1_std
      value: 1.9409534012355463
    - type: nauc_mrr_at_20_diff1
      value: 15.313905861533556
    - type: nauc_mrr_at_20_max
      value: 11.066794178767129
    - type: nauc_mrr_at_20_std
      value: 4.8481490714706545
    - type: nauc_mrr_at_3_diff1
      value: 17.61095753806274
    - type: nauc_mrr_at_3_max
      value: 10.366089044859502
    - type: nauc_mrr_at_3_std
      value: 4.49354511499649
    - type: nauc_mrr_at_5_diff1
      value: 16.108630589516295
    - type: nauc_mrr_at_5_max
      value: 11.240089407667481
    - type: nauc_mrr_at_5_std
      value: 4.872629531537418
    - type: nauc_ndcg_at_1000_diff1
      value: 12.77738687769916
    - type: nauc_ndcg_at_1000_max
      value: 12.549168176821333
    - type: nauc_ndcg_at_1000_std
      value: 8.144261457560836
    - type: nauc_ndcg_at_100_diff1
      value: 12.366782181161682
    - type: nauc_ndcg_at_100_max
      value: 10.925739246857757
    - type: nauc_ndcg_at_100_std
      value: 6.689593820129615
    - type: nauc_ndcg_at_10_diff1
      value: 10.27658665690359
    - type: nauc_ndcg_at_10_max
      value: 10.668336952263012
    - type: nauc_ndcg_at_10_std
      value: 4.4421604549442
    - type: nauc_ndcg_at_1_diff1
      value: 24.902916052765633
    - type: nauc_ndcg_at_1_max
      value: 9.457290628689096
    - type: nauc_ndcg_at_1_std
      value: 1.9409534012355463
    - type: nauc_ndcg_at_20_diff1
      value: 11.717938489930228
    - type: nauc_ndcg_at_20_max
      value: 11.406968575351918
    - type: nauc_ndcg_at_20_std
      value: 5.768402464744413
    - type: nauc_ndcg_at_3_diff1
      value: 15.60942938229517
    - type: nauc_ndcg_at_3_max
      value: 9.483164984948264
    - type: nauc_ndcg_at_3_std
      value: 5.000018271561521
    - type: nauc_ndcg_at_5_diff1
      value: 12.132383932726349
    - type: nauc_ndcg_at_5_max
      value: 10.951658963146887
    - type: nauc_ndcg_at_5_std
      value: 4.578775711947606
    - type: nauc_precision_at_1000_diff1
      value: 1.525064048392746
    - type: nauc_precision_at_1000_max
      value: 10.570950987477232
    - type: nauc_precision_at_1000_std
      value: 8.66408675090561
    - type: nauc_precision_at_100_diff1
      value: 5.8643392920971955
    - type: nauc_precision_at_100_max
      value: 9.250679060906934
    - type: nauc_precision_at_100_std
      value: 8.348394285666982
    - type: nauc_precision_at_10_diff1
      value: 1.2622567591326674
    - type: nauc_precision_at_10_max
      value: 12.089966028497381
    - type: nauc_precision_at_10_std
      value: 6.265235180800634
    - type: nauc_precision_at_1_diff1
      value: 24.902916052765633
    - type: nauc_precision_at_1_max
      value: 9.457290628689096
    - type: nauc_precision_at_1_std
      value: 1.9409534012355463
    - type: nauc_precision_at_20_diff1
      value: 6.213057335341744
    - type: nauc_precision_at_20_max
      value: 11.784391613266772
    - type: nauc_precision_at_20_std
      value: 8.07175929908232
    - type: nauc_precision_at_3_diff1
      value: 11.885541011959809
    - type: nauc_precision_at_3_max
      value: 11.281984764236645
    - type: nauc_precision_at_3_std
      value: 5.6489926433109945
    - type: nauc_precision_at_5_diff1
      value: 5.418248228174057
    - type: nauc_precision_at_5_max
      value: 13.026748231164703
    - type: nauc_precision_at_5_std
      value: 4.918677235989275
    - type: nauc_recall_at_1000_diff1
      value: 11.610656629742031
    - type: nauc_recall_at_1000_max
      value: 19.460310253620186
    - type: nauc_recall_at_1000_std
      value: 20.248445144276527
    - type: nauc_recall_at_100_diff1
      value: 9.369087091417065
    - type: nauc_recall_at_100_max
      value: 11.173394514490449
    - type: nauc_recall_at_100_std
      value: 10.671999236699662
    - type: nauc_recall_at_10_diff1
      value: 3.8578840249529693
    - type: nauc_recall_at_10_max
      value: 12.090794523545023
    - type: nauc_recall_at_10_std
      value: 3.9686816569682257
    - type: nauc_recall_at_1_diff1
      value: 24.022346561431156
    - type: nauc_recall_at_1_max
      value: 7.841750008744963
    - type: nauc_recall_at_1_std
      value: 3.3784810578492785
    - type: nauc_recall_at_20_diff1
      value: 7.7817049051114955
    - type: nauc_recall_at_20_max
      value: 13.7356823026863
    - type: nauc_recall_at_20_std
      value: 7.5011443092451575
    - type: nauc_recall_at_3_diff1
      value: 11.867264801692084
    - type: nauc_recall_at_3_max
      value: 9.772515269996166
    - type: nauc_recall_at_3_std
      value: 5.649898224902724
    - type: nauc_recall_at_5_diff1
      value: 6.7943445242888085
    - type: nauc_recall_at_5_max
      value: 13.134954628949208
    - type: nauc_recall_at_5_std
      value: 4.8683311340579465
    - type: ndcg_at_1
      value: 8.581999999999999
    - type: ndcg_at_10
      value: 13.624
    - type: ndcg_at_100
      value: 18.361
    - type: ndcg_at_1000
      value: 22.017
    - type: ndcg_at_20
      value: 15.040000000000001
    - type: ndcg_at_3
      value: 10.735
    - type: ndcg_at_5
      value: 12.123000000000001
    - type: precision_at_1
      value: 8.581999999999999
    - type: precision_at_10
      value: 2.637
    - type: precision_at_100
      value: 0.59
    - type: precision_at_1000
      value: 0.105
    - type: precision_at_20
      value: 1.704
    - type: precision_at_3
      value: 5.1
    - type: precision_at_5
      value: 4.005
    - type: recall_at_1
      value: 7.0809999999999995
    - type: recall_at_10
      value: 20.022000000000002
    - type: recall_at_100
      value: 41.921
    - type: recall_at_1000
      value: 68.60199999999999
    - type: recall_at_20
      value: 25.156
    - type: recall_at_3
      value: 12.432
    - type: recall_at_5
      value: 15.628
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackPhysicsRetrieval (default)
      revision: 79531abbd1fb92d06c6d6315a0cbbbf5bb247ea4
      split: test
      type: mteb/cqadupstack-physics
    metrics:
    - type: main_score
      value: 28.1
    - type: map_at_1
      value: 17.408
    - type: map_at_10
      value: 23.634
    - type: map_at_100
      value: 24.759999999999998
    - type: map_at_1000
      value: 24.901
    - type: map_at_20
      value: 24.144
    - type: map_at_3
      value: 21.317
    - type: map_at_5
      value: 22.615
    - type: mrr_at_1
      value: 21.751684311838307
    - type: mrr_at_10
      value: 28.205463128466008
    - type: mrr_at_100
      value: 29.135361539963515
    - type: mrr_at_1000
      value: 29.21658309312368
    - type: mrr_at_20
      value: 28.721775483050898
    - type: mrr_at_3
      value: 25.794032723772865
    - type: mrr_at_5
      value: 27.175168431183817
    - type: nauc_map_at_1000_diff1
      value: 43.260365034580076
    - type: nauc_map_at_1000_max
      value: 25.979886910558747
    - type: nauc_map_at_1000_std
      value: -2.0849267220215113
    - type: nauc_map_at_100_diff1
      value: 43.224870619497985
    - type: nauc_map_at_100_max
      value: 25.88024877542956
    - type: nauc_map_at_100_std
      value: -2.21391594997994
    - type: nauc_map_at_10_diff1
      value: 43.47446736665736
    - type: nauc_map_at_10_max
      value: 25.97259760761233
    - type: nauc_map_at_10_std
      value: -2.7682768946830407
    - type: nauc_map_at_1_diff1
      value: 51.23237029644942
    - type: nauc_map_at_1_max
      value: 29.6416845733838
    - type: nauc_map_at_1_std
      value: -2.806567544030298
    - type: nauc_map_at_20_diff1
      value: 43.164927055048096
    - type: nauc_map_at_20_max
      value: 25.79620371040526
    - type: nauc_map_at_20_std
      value: -2.527474807557985
    - type: nauc_map_at_3_diff1
      value: 45.16292828974055
    - type: nauc_map_at_3_max
      value: 26.192526759218914
    - type: nauc_map_at_3_std
      value: -3.258122441754642
    - type: nauc_map_at_5_diff1
      value: 44.27022371461212
    - type: nauc_map_at_5_max
      value: 25.986943086976233
    - type: nauc_map_at_5_std
      value: -2.96882589367969
    - type: nauc_mrr_at_1000_diff1
      value: 40.76319304785522
    - type: nauc_mrr_at_1000_max
      value: 26.528028493585577
    - type: nauc_mrr_at_1000_std
      value: 0.5361028661180448
    - type: nauc_mrr_at_100_diff1
      value: 40.72595239434217
    - type: nauc_mrr_at_100_max
      value: 26.47864694368845
    - type: nauc_mrr_at_100_std
      value: 0.5015817550912431
    - type: nauc_mrr_at_10_diff1
      value: 40.95782826805433
    - type: nauc_mrr_at_10_max
      value: 26.72257097851632
    - type: nauc_mrr_at_10_std
      value: 0.3297285535383387
    - type: nauc_mrr_at_1_diff1
      value: 47.72594645013734
    - type: nauc_mrr_at_1_max
      value: 30.394324110030567
    - type: nauc_mrr_at_1_std
      value: 1.4420412083862328
    - type: nauc_mrr_at_20_diff1
      value: 40.639149396407674
    - type: nauc_mrr_at_20_max
      value: 26.4962504824028
    - type: nauc_mrr_at_20_std
      value: 0.41981984468310246
    - type: nauc_mrr_at_3_diff1
      value: 42.149504777302724
    - type: nauc_mrr_at_3_max
      value: 27.027528584859734
    - type: nauc_mrr_at_3_std
      value: -0.3484158715300914
    - type: nauc_mrr_at_5_diff1
      value: 41.395178216037635
    - type: nauc_mrr_at_5_max
      value: 27.06751242405021
    - type: nauc_mrr_at_5_std
      value: 0.24717610402157794
    - type: nauc_ndcg_at_1000_diff1
      value: 39.575508159078474
    - type: nauc_ndcg_at_1000_max
      value: 25.568538008813128
    - type: nauc_ndcg_at_1000_std
      value: 2.0843318101121113
    - type: nauc_ndcg_at_100_diff1
      value: 38.84505152672922
    - type: nauc_ndcg_at_100_max
      value: 24.164287066890424
    - type: nauc_ndcg_at_100_std
      value: 0.4704663117394464
    - type: nauc_ndcg_at_10_diff1
      value: 39.59921895637892
    - type: nauc_ndcg_at_10_max
      value: 24.59345472310171
    - type: nauc_ndcg_at_10_std
      value: -1.8816000573302147
    - type: nauc_ndcg_at_1_diff1
      value: 47.72594645013734
    - type: nauc_ndcg_at_1_max
      value: 30.394324110030567
    - type: nauc_ndcg_at_1_std
      value: 1.4420412083862328
    - type: nauc_ndcg_at_20_diff1
      value: 38.49214533778004
    - type: nauc_ndcg_at_20_max
      value: 23.872891791896738
    - type: nauc_ndcg_at_20_std
      value: -1.2296334118794574
    - type: nauc_ndcg_at_3_diff1
      value: 42.42673245467605
    - type: nauc_ndcg_at_3_max
      value: 26.047493631745866
    - type: nauc_ndcg_at_3_std
      value: -1.9063204994807348
    - type: nauc_ndcg_at_5_diff1
      value: 41.130666312853634
    - type: nauc_ndcg_at_5_max
      value: 25.366500049103458
    - type: nauc_ndcg_at_5_std
      value: -1.8421163435361618
    - type: nauc_precision_at_1000_diff1
      value: 1.7292134029280288
    - type: nauc_precision_at_1000_max
      value: 13.882127060898435
    - type: nauc_precision_at_1000_std
      value: 21.261694377521952
    - type: nauc_precision_at_100_diff1
      value: 11.75338508774606
    - type: nauc_precision_at_100_max
      value: 17.089840857788904
    - type: nauc_precision_at_100_std
      value: 15.621459805832355
    - type: nauc_precision_at_10_diff1
      value: 23.41970845165901
    - type: nauc_precision_at_10_max
      value: 23.276769802630838
    - type: nauc_precision_at_10_std
      value: 6.3952347911663345
    - type: nauc_precision_at_1_diff1
      value: 47.72594645013734
    - type: nauc_precision_at_1_max
      value: 30.394324110030567
    - type: nauc_precision_at_1_std
      value: 1.4420412083862328
    - type: nauc_precision_at_20_diff1
      value: 17.873164002378125
    - type: nauc_precision_at_20_max
      value: 19.92501795496989
    - type: nauc_precision_at_20_std
      value: 9.199279241356155
    - type: nauc_precision_at_3_diff1
      value: 33.646829737006726
    - type: nauc_precision_at_3_max
      value: 25.281870315662353
    - type: nauc_precision_at_3_std
      value: 1.8786825941907552
    - type: nauc_precision_at_5_diff1
      value: 29.609102130780364
    - type: nauc_precision_at_5_max
      value: 25.096715780488978
    - type: nauc_precision_at_5_std
      value: 3.9900430759799645
    - type: nauc_recall_at_1000_diff1
      value: 23.069295215730058
    - type: nauc_recall_at_1000_max
      value: 20.572528914896765
    - type: nauc_recall_at_1000_std
      value: 29.83709673991498
    - type: nauc_recall_at_100_diff1
      value: 23.33784906640149
    - type: nauc_recall_at_100_max
      value: 13.56321944922501
    - type: nauc_recall_at_100_std
      value: 7.46189623877132
    - type: nauc_recall_at_10_diff1
      value: 29.226182488961662
    - type: nauc_recall_at_10_max
      value: 18.3684951121155
    - type: nauc_recall_at_10_std
      value: -2.5415354865089634
    - type: nauc_recall_at_1_diff1
      value: 51.23237029644942
    - type: nauc_recall_at_1_max
      value: 29.6416845733838
    - type: nauc_recall_at_1_std
      value: -2.806567544030298
    - type: nauc_recall_at_20_diff1
      value: 24.982279789513655
    - type: nauc_recall_at_20_max
      value: 15.35005827725592
    - type: nauc_recall_at_20_std
      value: -0.4272837479647023
    - type: nauc_recall_at_3_diff1
      value: 37.0322476528056
    - type: nauc_recall_at_3_max
      value: 21.523706479074505
    - type: nauc_recall_at_3_std
      value: -3.6419367768075075
    - type: nauc_recall_at_5_diff1
      value: 33.729365708218175
    - type: nauc_recall_at_5_max
      value: 20.29944173157368
    - type: nauc_recall_at_5_std
      value: -2.9020859696575236
    - type: ndcg_at_1
      value: 21.752
    - type: ndcg_at_10
      value: 28.1
    - type: ndcg_at_100
      value: 33.794000000000004
    - type: ndcg_at_1000
      value: 36.83
    - type: ndcg_at_20
      value: 29.843999999999998
    - type: ndcg_at_3
      value: 23.990000000000002
    - type: ndcg_at_5
      value: 25.94
    - type: precision_at_1
      value: 21.752
    - type: precision_at_10
      value: 5.207
    - type: precision_at_100
      value: 0.98
    - type: precision_at_1000
      value: 0.14300000000000002
    - type: precision_at_20
      value: 3.152
    - type: precision_at_3
      value: 11.229
    - type: precision_at_5
      value: 8.315999999999999
    - type: recall_at_1
      value: 17.408
    - type: recall_at_10
      value: 37.165
    - type: recall_at_100
      value: 62.651
    - type: recall_at_1000
      value: 83.46900000000001
    - type: recall_at_20
      value: 43.446
    - type: recall_at_3
      value: 25.6
    - type: recall_at_5
      value: 30.654999999999998
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackProgrammersRetrieval (default)
      revision: 6184bc1440d2dbc7612be22b50686b8826d22b32
      split: test
      type: mteb/cqadupstack-programmers
    metrics:
    - type: main_score
      value: 23.433
    - type: map_at_1
      value: 13.699
    - type: map_at_10
      value: 19.466
    - type: map_at_100
      value: 20.583000000000002
    - type: map_at_1000
      value: 20.724999999999998
    - type: map_at_20
      value: 20.043
    - type: map_at_3
      value: 17.462
    - type: map_at_5
      value: 18.529
    - type: mrr_at_1
      value: 16.32420091324201
    - type: mrr_at_10
      value: 22.843598245995487
    - type: mrr_at_100
      value: 23.805814788577347
    - type: mrr_at_1000
      value: 23.893595459117634
    - type: mrr_at_20
      value: 23.364083699697467
    - type: mrr_at_3
      value: 20.71917808219179
    - type: mrr_at_5
      value: 21.923515981735154
    - type: nauc_map_at_1000_diff1
      value: 35.29908176677659
    - type: nauc_map_at_1000_max
      value: 21.19659387825901
    - type: nauc_map_at_1000_std
      value: -1.1800470644368641
    - type: nauc_map_at_100_diff1
      value: 35.25437342956425
    - type: nauc_map_at_100_max
      value: 21.158048137454784
    - type: nauc_map_at_100_std
      value: -1.2790123198148806
    - type: nauc_map_at_10_diff1
      value: 35.50950328919451
    - type: nauc_map_at_10_max
      value: 20.348484824359996
    - type: nauc_map_at_10_std
      value: -2.4603394968686154
    - type: nauc_map_at_1_diff1
      value: 41.10727065397603
    - type: nauc_map_at_1_max
      value: 22.543501102453543
    - type: nauc_map_at_1_std
      value: -2.7943381702992465
    - type: nauc_map_at_20_diff1
      value: 35.19029596236487
    - type: nauc_map_at_20_max
      value: 20.87279038260115
    - type: nauc_map_at_20_std
      value: -1.82363477470446
    - type: nauc_map_at_3_diff1
      value: 36.02030150485208
    - type: nauc_map_at_3_max
      value: 20.38357498006346
    - type: nauc_map_at_3_std
      value: -2.837404356814094
    - type: nauc_map_at_5_diff1
      value: 35.8201348144381
    - type: nauc_map_at_5_max
      value: 20.38035108882887
    - type: nauc_map_at_5_std
      value: -2.7002422977828346
    - type: nauc_mrr_at_1000_diff1
      value: 35.04099226290557
    - type: nauc_mrr_at_1000_max
      value: 21.57427001290516
    - type: nauc_mrr_at_1000_std
      value: 0.3715347702019008
    - type: nauc_mrr_at_100_diff1
      value: 35.00428106691662
    - type: nauc_mrr_at_100_max
      value: 21.56784751335325
    - type: nauc_mrr_at_100_std
      value: 0.3466863156775645
    - type: nauc_mrr_at_10_diff1
      value: 35.19156170525377
    - type: nauc_mrr_at_10_max
      value: 21.106207145262328
    - type: nauc_mrr_at_10_std
      value: -0.5031832849355399
    - type: nauc_mrr_at_1_diff1
      value: 40.722261443790906
    - type: nauc_mrr_at_1_max
      value: 24.15698098634036
    - type: nauc_mrr_at_1_std
      value: -0.30639756688939146
    - type: nauc_mrr_at_20_diff1
      value: 35.00790634012167
    - type: nauc_mrr_at_20_max
      value: 21.383803803042724
    - type: nauc_mrr_at_20_std
      value: -0.05435467437352896
    - type: nauc_mrr_at_3_diff1
      value: 36.289465305244846
    - type: nauc_mrr_at_3_max
      value: 22.291792865731253
    - type: nauc_mrr_at_3_std
      value: -0.1601560688322784
    - type: nauc_mrr_at_5_diff1
      value: 36.03061040405196
    - type: nauc_mrr_at_5_max
      value: 21.4330773038141
    - type: nauc_mrr_at_5_std
      value: -0.36308819446465274
    - type: nauc_ndcg_at_1000_diff1
      value: 33.232430146174295
    - type: nauc_ndcg_at_1000_max
      value: 22.983696106878117
    - type: nauc_ndcg_at_1000_std
      value: 4.640830565692821
    - type: nauc_ndcg_at_100_diff1
      value: 32.50338054067435
    - type: nauc_ndcg_at_100_max
      value: 22.189511219317435
    - type: nauc_ndcg_at_100_std
      value: 3.105218998038352
    - type: nauc_ndcg_at_10_diff1
      value: 33.32318612218884
    - type: nauc_ndcg_at_10_max
      value: 19.56436410436655
    - type: nauc_ndcg_at_10_std
      value: -1.8344884585445502
    - type: nauc_ndcg_at_1_diff1
      value: 40.722261443790906
    - type: nauc_ndcg_at_1_max
      value: 24.15698098634036
    - type: nauc_ndcg_at_1_std
      value: -0.30639756688939146
    - type: nauc_ndcg_at_20_diff1
      value: 32.347401402734775
    - type: nauc_ndcg_at_20_max
      value: 20.83380671662441
    - type: nauc_ndcg_at_20_std
      value: 0.016563433585529974
    - type: nauc_ndcg_at_3_diff1
      value: 35.04188810519525
    - type: nauc_ndcg_at_3_max
      value: 20.764019978598487
    - type: nauc_ndcg_at_3_std
      value: -1.594528012527463
    - type: nauc_ndcg_at_5_diff1
      value: 34.43943202369672
    - type: nauc_ndcg_at_5_max
      value: 20.106254608612055
    - type: nauc_ndcg_at_5_std
      value: -1.8594865842617228
    - type: nauc_precision_at_1000_diff1
      value: 7.256129492861672
    - type: nauc_precision_at_1000_max
      value: 11.996188955211178
    - type: nauc_precision_at_1000_std
      value: 13.821279312799087
    - type: nauc_precision_at_100_diff1
      value: 16.06033204193287
    - type: nauc_precision_at_100_max
      value: 21.298571657566136
    - type: nauc_precision_at_100_std
      value: 16.33488809216804
    - type: nauc_precision_at_10_diff1
      value: 25.542062522295577
    - type: nauc_precision_at_10_max
      value: 20.011563586461563
    - type: nauc_precision_at_10_std
      value: 3.0291709497281682
    - type: nauc_precision_at_1_diff1
      value: 40.722261443790906
    - type: nauc_precision_at_1_max
      value: 24.15698098634036
    - type: nauc_precision_at_1_std
      value: -0.30639756688939146
    - type: nauc_precision_at_20_diff1
      value: 21.545331269787848
    - type: nauc_precision_at_20_max
      value: 21.328836807337613
    - type: nauc_precision_at_20_std
      value: 6.749273704342807
    - type: nauc_precision_at_3_diff1
      value: 31.74754387933826
    - type: nauc_precision_at_3_max
      value: 22.00131054032921
    - type: nauc_precision_at_3_std
      value: 0.8933994096049079
    - type: nauc_precision_at_5_diff1
      value: 30.086127422078313
    - type: nauc_precision_at_5_max
      value: 20.07469952891432
    - type: nauc_precision_at_5_std
      value: 0.16970208205211193
    - type: nauc_recall_at_1000_diff1
      value: 20.943658951883773
    - type: nauc_recall_at_1000_max
      value: 33.25768046293579
    - type: nauc_recall_at_1000_std
      value: 41.359796251893364
    - type: nauc_recall_at_100_diff1
      value: 21.484586350505037
    - type: nauc_recall_at_100_max
      value: 22.854675507253802
    - type: nauc_recall_at_100_std
      value: 16.02406263632089
    - type: nauc_recall_at_10_diff1
      value: 27.194572601872668
    - type: nauc_recall_at_10_max
      value: 16.06431177414546
    - type: nauc_recall_at_10_std
      value: -1.723302447498358
    - type: nauc_recall_at_1_diff1
      value: 41.10727065397603
    - type: nauc_recall_at_1_max
      value: 22.543501102453543
    - type: nauc_recall_at_1_std
      value: -2.7943381702992465
    - type: nauc_recall_at_20_diff1
      value: 23.584838236915115
    - type: nauc_recall_at_20_max
      value: 19.41983426995758
    - type: nauc_recall_at_20_std
      value: 3.986703252775787
    - type: nauc_recall_at_3_diff1
      value: 30.56479090838521
    - type: nauc_recall_at_3_max
      value: 17.872434147655504
    - type: nauc_recall_at_3_std
      value: -2.5457977048929803
    - type: nauc_recall_at_5_diff1
      value: 29.89579939854362
    - type: nauc_recall_at_5_max
      value: 17.285994867348798
    - type: nauc_recall_at_5_std
      value: -2.0188171694818413
    - type: ndcg_at_1
      value: 16.323999999999998
    - type: ndcg_at_10
      value: 23.433
    - type: ndcg_at_100
      value: 29.032000000000004
    - type: ndcg_at_1000
      value: 32.389
    - type: ndcg_at_20
      value: 25.369999999999997
    - type: ndcg_at_3
      value: 19.661
    - type: ndcg_at_5
      value: 21.369
    - type: precision_at_1
      value: 16.323999999999998
    - type: precision_at_10
      value: 4.543
    - type: precision_at_100
      value: 0.885
    - type: precision_at_1000
      value: 0.134
    - type: precision_at_20
      value: 2.8369999999999997
    - type: precision_at_3
      value: 9.399000000000001
    - type: precision_at_5
      value: 7.055
    - type: recall_at_1
      value: 13.699
    - type: recall_at_10
      value: 31.89
    - type: recall_at_100
      value: 56.785
    - type: recall_at_1000
      value: 80.697
    - type: recall_at_20
      value: 38.838
    - type: recall_at_3
      value: 21.813
    - type: recall_at_5
      value: 25.967000000000002
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackRetrieval (default)
      revision: CQADupstackRetrieval_is_a_combined_dataset
      split: test
      type: CQADupstackRetrieval_is_a_combined_dataset
    metrics:
    - type: main_score
      value: 23.69391666666667
    - type: ndcg_at_10
      value: 23.69391666666667
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackStatsRetrieval (default)
      revision: 65ac3a16b8e91f9cee4c9828cc7c335575432a2a
      split: test
      type: mteb/cqadupstack-stats
    metrics:
    - type: main_score
      value: 18.384
    - type: map_at_1
      value: 11.806
    - type: map_at_10
      value: 15.659
    - type: map_at_100
      value: 16.497999999999998
    - type: map_at_1000
      value: 16.595
    - type: map_at_20
      value: 16.18
    - type: map_at_3
      value: 14.137
    - type: map_at_5
      value: 14.956
    - type: mrr_at_1
      value: 13.957055214723926
    - type: mrr_at_10
      value: 17.797679910409965
    - type: mrr_at_100
      value: 18.634105330873982
    - type: mrr_at_1000
      value: 18.723872119576956
    - type: mrr_at_20
      value: 18.315768640080353
    - type: mrr_at_3
      value: 16.308793456032713
    - type: mrr_at_5
      value: 17.05265848670756
    - type: nauc_map_at_1000_diff1
      value: 24.451360941385527
    - type: nauc_map_at_1000_max
      value: 9.703731227062287
    - type: nauc_map_at_1000_std
      value: 1.012478774475079
    - type: nauc_map_at_100_diff1
      value: 24.45541062900218
    - type: nauc_map_at_100_max
      value: 9.649455487883655
    - type: nauc_map_at_100_std
      value: 0.910681103488745
    - type: nauc_map_at_10_diff1
      value: 24.623702264364173
    - type: nauc_map_at_10_max
      value: 9.96456196295675
    - type: nauc_map_at_10_std
      value: 0.6883064213253189
    - type: nauc_map_at_1_diff1
      value: 24.463253990774003
    - type: nauc_map_at_1_max
      value: 8.44516445733758
    - type: nauc_map_at_1_std
      value: -2.705228609307227
    - type: nauc_map_at_20_diff1
      value: 24.41807093882515
    - type: nauc_map_at_20_max
      value: 9.73079819255772
    - type: nauc_map_at_20_std
      value: 0.8307302299269684
    - type: nauc_map_at_3_diff1
      value: 24.647027786393476
    - type: nauc_map_at_3_max
      value: 10.158348799557
    - type: nauc_map_at_3_std
      value: 0.6596938602041736
    - type: nauc_map_at_5_diff1
      value: 24.757624011943456
    - type: nauc_map_at_5_max
      value: 10.375814716590098
    - type: nauc_map_at_5_std
      value: 0.048075053740994585
    - type: nauc_mrr_at_1000_diff1
      value: 26.62518268407276
    - type: nauc_mrr_at_1000_max
      value: 13.855993519763175
    - type: nauc_mrr_at_1000_std
      value: 4.62905337450327
    - type: nauc_mrr_at_100_diff1
      value: 26.607592560385196
    - type: nauc_mrr_at_100_max
      value: 13.811494487812942
    - type: nauc_mrr_at_100_std
      value: 4.555702211448958
    - type: nauc_mrr_at_10_diff1
      value: 26.885862463851208
    - type: nauc_mrr_at_10_max
      value: 14.333094390691404
    - type: nauc_mrr_at_10_std
      value: 4.692620223972844
    - type: nauc_mrr_at_1_diff1
      value: 29.456424929748838
    - type: nauc_mrr_at_1_max
      value: 14.906357362630688
    - type: nauc_mrr_at_1_std
      value: 2.4543413134245498
    - type: nauc_mrr_at_20_diff1
      value: 26.635584926784205
    - type: nauc_mrr_at_20_max
      value: 13.983672186253978
    - type: nauc_mrr_at_20_std
      value: 4.533797671279914
    - type: nauc_mrr_at_3_diff1
      value: 27.257431270887867
    - type: nauc_mrr_at_3_max
      value: 14.51038573428384
    - type: nauc_mrr_at_3_std
      value: 4.677541680669749
    - type: nauc_mrr_at_5_diff1
      value: 27.308407985683008
    - type: nauc_mrr_at_5_max
      value: 14.725017635611035
    - type: nauc_mrr_at_5_std
      value: 4.069145623758021
    - type: nauc_ndcg_at_1000_diff1
      value: 23.87595648875096
    - type: nauc_ndcg_at_1000_max
      value: 10.790320900447414
    - type: nauc_ndcg_at_1000_std
      value: 5.82775131929226
    - type: nauc_ndcg_at_100_diff1
      value: 23.80489864326311
    - type: nauc_ndcg_at_100_max
      value: 9.542909613337207
    - type: nauc_ndcg_at_100_std
      value: 3.514583939812324
    - type: nauc_ndcg_at_10_diff1
      value: 24.468523939453547
    - type: nauc_ndcg_at_10_max
      value: 10.65620836221067
    - type: nauc_ndcg_at_10_std
      value: 2.5824019348755596
    - type: nauc_ndcg_at_1_diff1
      value: 29.456424929748838
    - type: nauc_ndcg_at_1_max
      value: 14.906357362630688
    - type: nauc_ndcg_at_1_std
      value: 2.4543413134245498
    - type: nauc_ndcg_at_20_diff1
      value: 23.90499485855613
    - type: nauc_ndcg_at_20_max
      value: 9.931714836248881
    - type: nauc_ndcg_at_20_std
      value: 2.823910728207098
    - type: nauc_ndcg_at_3_diff1
      value: 25.1439714946535
    - type: nauc_ndcg_at_3_max
      value: 11.314410735026595
    - type: nauc_ndcg_at_3_std
      value: 2.6451314305581435
    - type: nauc_ndcg_at_5_diff1
      value: 25.002955818196547
    - type: nauc_ndcg_at_5_max
      value: 11.626656177248531
    - type: nauc_ndcg_at_5_std
      value: 1.2826883242759335
    - type: nauc_precision_at_1000_diff1
      value: 17.91291822870243
    - type: nauc_precision_at_1000_max
      value: 16.76675856170149
    - type: nauc_precision_at_1000_std
      value: 21.28676061856954
    - type: nauc_precision_at_100_diff1
      value: 22.65384922327202
    - type: nauc_precision_at_100_max
      value: 14.035389695379298
    - type: nauc_precision_at_100_std
      value: 13.11748433352797
    - type: nauc_precision_at_10_diff1
      value: 26.851662967794372
    - type: nauc_precision_at_10_max
      value: 15.712260548565924
    - type: nauc_precision_at_10_std
      value: 9.920191712601452
    - type: nauc_precision_at_1_diff1
      value: 29.456424929748838
    - type: nauc_precision_at_1_max
      value: 14.906357362630688
    - type: nauc_precision_at_1_std
      value: 2.4543413134245498
    - type: nauc_precision_at_20_diff1
      value: 24.220224327836686
    - type: nauc_precision_at_20_max
      value: 14.588302868445297
    - type: nauc_precision_at_20_std
      value: 10.92824540303324
    - type: nauc_precision_at_3_diff1
      value: 28.488577734428645
    - type: nauc_precision_at_3_max
      value: 15.98246205564231
    - type: nauc_precision_at_3_std
      value: 7.303904258068353
    - type: nauc_precision_at_5_diff1
      value: 28.939989997482773
    - type: nauc_precision_at_5_max
      value: 17.041774400394182
    - type: nauc_precision_at_5_std
      value: 5.509812446342075
    - type: nauc_recall_at_1000_diff1
      value: 18.267969017984136
    - type: nauc_recall_at_1000_max
      value: 11.388634142238113
    - type: nauc_recall_at_1000_std
      value: 23.91731245454964
    - type: nauc_recall_at_100_diff1
      value: 19.151833799379745
    - type: nauc_recall_at_100_max
      value: 4.557136315212161
    - type: nauc_recall_at_100_std
      value: 7.149764457086401
    - type: nauc_recall_at_10_diff1
      value: 21.410038364719394
    - type: nauc_recall_at_10_max
      value: 8.068322780045472
    - type: nauc_recall_at_10_std
      value: 3.7037571191716747
    - type: nauc_recall_at_1_diff1
      value: 24.463253990774003
    - type: nauc_recall_at_1_max
      value: 8.44516445733758
    - type: nauc_recall_at_1_std
      value: -2.705228609307227
    - type: nauc_recall_at_20_diff1
      value: 19.90729636016057
    - type: nauc_recall_at_20_max
      value: 5.978272392594185
    - type: nauc_recall_at_20_std
      value: 4.320061234796331
    - type: nauc_recall_at_3_diff1
      value: 22.871302219459302
    - type: nauc_recall_at_3_max
      value: 10.472443448415733
    - type: nauc_recall_at_3_std
      value: 3.013314639210661
    - type: nauc_recall_at_5_diff1
      value: 22.054793329886756
    - type: nauc_recall_at_5_max
      value: 10.796778960647508
    - type: nauc_recall_at_5_std
      value: 0.814500718437171
    - type: ndcg_at_1
      value: 13.957
    - type: ndcg_at_10
      value: 18.384
    - type: ndcg_at_100
      value: 22.607
    - type: ndcg_at_1000
      value: 25.466
    - type: ndcg_at_20
      value: 20.23
    - type: ndcg_at_3
      value: 15.527
    - type: ndcg_at_5
      value: 16.802
    - type: precision_at_1
      value: 13.957
    - type: precision_at_10
      value: 3.0669999999999997
    - type: precision_at_100
      value: 0.555
    - type: precision_at_1000
      value: 0.087
    - type: precision_at_20
      value: 1.9709999999999999
    - type: precision_at_3
      value: 6.800000000000001
    - type: precision_at_5
      value: 4.939
    - type: recall_at_1
      value: 11.806
    - type: recall_at_10
      value: 24.837999999999997
    - type: recall_at_100
      value: 44.181
    - type: recall_at_1000
      value: 65.81099999999999
    - type: recall_at_20
      value: 31.863000000000003
    - type: recall_at_3
      value: 16.956
    - type: recall_at_5
      value: 20.112
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackTexRetrieval (default)
      revision: 46989137a86843e03a6195de44b09deda022eec7
      split: test
      type: mteb/cqadupstack-tex
    metrics:
    - type: main_score
      value: 13.447000000000001
    - type: map_at_1
      value: 7.4990000000000006
    - type: map_at_10
      value: 10.988000000000001
    - type: map_at_100
      value: 11.698
    - type: map_at_1000
      value: 11.822000000000001
    - type: map_at_20
      value: 11.349
    - type: map_at_3
      value: 9.777
    - type: map_at_5
      value: 10.497
    - type: mrr_at_1
      value: 9.015829318651067
    - type: mrr_at_10
      value: 13.109794732299878
    - type: mrr_at_100
      value: 13.868221537379819
    - type: mrr_at_1000
      value: 13.968069797611808
    - type: mrr_at_20
      value: 13.519507570996245
    - type: mrr_at_3
      value: 11.75728378068364
    - type: mrr_at_5
      value: 12.557352603808189
    - type: nauc_map_at_1000_diff1
      value: 30.567002294769587
    - type: nauc_map_at_1000_max
      value: 15.798512312830034
    - type: nauc_map_at_1000_std
      value: -1.0859390117945695
    - type: nauc_map_at_100_diff1
      value: 30.587286610445663
    - type: nauc_map_at_100_max
      value: 15.73430549325338
    - type: nauc_map_at_100_std
      value: -1.1937200115265094
    - type: nauc_map_at_10_diff1
      value: 31.408895433670995
    - type: nauc_map_at_10_max
      value: 15.77429808454488
    - type: nauc_map_at_10_std
      value: -1.6732090098666632
    - type: nauc_map_at_1_diff1
      value: 40.57399794385126
    - type: nauc_map_at_1_max
      value: 16.149171625208833
    - type: nauc_map_at_1_std
      value: -2.4218080519307703
    - type: nauc_map_at_20_diff1
      value: 30.8482877273114
    - type: nauc_map_at_20_max
      value: 15.720632965606299
    - type: nauc_map_at_20_std
      value: -1.5062672633905416
    - type: nauc_map_at_3_diff1
      value: 33.52569401786841
    - type: nauc_map_at_3_max
      value: 16.615076553444595
    - type: nauc_map_at_3_std
      value: -1.6384914831039104
    - type: nauc_map_at_5_diff1
      value: 32.329365367117866
    - type: nauc_map_at_5_max
      value: 16.216667669611805
    - type: nauc_map_at_5_std
      value: -1.6533435358995812
    - type: nauc_mrr_at_1000_diff1
      value: 29.31052121532543
    - type: nauc_mrr_at_1000_max
      value: 17.53494423521252
    - type: nauc_mrr_at_1000_std
      value: 0.5840623134394628
    - type: nauc_mrr_at_100_diff1
      value: 29.323160649455303
    - type: nauc_mrr_at_100_max
      value: 17.529292065145697
    - type: nauc_mrr_at_100_std
      value: 0.5580584016696392
    - type: nauc_mrr_at_10_diff1
      value: 29.981137580131602
    - type: nauc_mrr_at_10_max
      value: 17.615794933598455
    - type: nauc_mrr_at_10_std
      value: 0.2123100820677504
    - type: nauc_mrr_at_1_diff1
      value: 38.54985300740823
    - type: nauc_mrr_at_1_max
      value: 18.110835976336595
    - type: nauc_mrr_at_1_std
      value: -1.7903397288683682
    - type: nauc_mrr_at_20_diff1
      value: 29.499330291362007
    - type: nauc_mrr_at_20_max
      value: 17.550155012432754
    - type: nauc_mrr_at_20_std
      value: 0.33244192880021073
    - type: nauc_mrr_at_3_diff1
      value: 31.78443905858544
    - type: nauc_mrr_at_3_max
      value: 18.622792547430922
    - type: nauc_mrr_at_3_std
      value: 0.10837491386435653
    - type: nauc_mrr_at_5_diff1
      value: 30.634162157096757
    - type: nauc_mrr_at_5_max
      value: 17.953005511330144
    - type: nauc_mrr_at_5_std
      value: 0.3011188554086528
    - type: nauc_ndcg_at_1000_diff1
      value: 24.927260803572327
    - type: nauc_ndcg_at_1000_max
      value: 15.735187217587246
    - type: nauc_ndcg_at_1000_std
      value: 2.70536509701587
    - type: nauc_ndcg_at_100_diff1
      value: 25.222134569922545
    - type: nauc_ndcg_at_100_max
      value: 15.010537382520306
    - type: nauc_ndcg_at_100_std
      value: 1.0428591825830975
    - type: nauc_ndcg_at_10_diff1
      value: 27.980571708839424
    - type: nauc_ndcg_at_10_max
      value: 15.362543479684104
    - type: nauc_ndcg_at_10_std
      value: -1.063042923474815
    - type: nauc_ndcg_at_1_diff1
      value: 38.54985300740823
    - type: nauc_ndcg_at_1_max
      value: 18.110835976336595
    - type: nauc_ndcg_at_1_std
      value: -1.7903397288683682
    - type: nauc_ndcg_at_20_diff1
      value: 26.49918565200889
    - type: nauc_ndcg_at_20_max
      value: 15.176785050941795
    - type: nauc_ndcg_at_20_std
      value: -0.5768848393065947
    - type: nauc_ndcg_at_3_diff1
      value: 31.274664013387905
    - type: nauc_ndcg_at_3_max
      value: 17.388908933640423
    - type: nauc_ndcg_at_3_std
      value: -0.6986711713763971
    - type: nauc_ndcg_at_5_diff1
      value: 29.55493575480121
    - type: nauc_ndcg_at_5_max
      value: 16.331948175175768
    - type: nauc_ndcg_at_5_std
      value: -0.7821520443168124
    - type: nauc_precision_at_1000_diff1
      value: 12.34617356216753
    - type: nauc_precision_at_1000_max
      value: 24.354588824888587
    - type: nauc_precision_at_1000_std
      value: 17.750963682106143
    - type: nauc_precision_at_100_diff1
      value: 14.963876212987392
    - type: nauc_precision_at_100_max
      value: 19.370472977151568
    - type: nauc_precision_at_100_std
      value: 9.52576847541998
    - type: nauc_precision_at_10_diff1
      value: 20.07061887015277
    - type: nauc_precision_at_10_max
      value: 16.71813468561834
    - type: nauc_precision_at_10_std
      value: 1.593454819519877
    - type: nauc_precision_at_1_diff1
      value: 38.54985300740823
    - type: nauc_precision_at_1_max
      value: 18.110835976336595
    - type: nauc_precision_at_1_std
      value: -1.7903397288683682
    - type: nauc_precision_at_20_diff1
      value: 16.740457192476917
    - type: nauc_precision_at_20_max
      value: 17.946474788321787
    - type: nauc_precision_at_20_std
      value: 3.9603159921284763
    - type: nauc_precision_at_3_diff1
      value: 26.55973349893991
    - type: nauc_precision_at_3_max
      value: 19.498328112386986
    - type: nauc_precision_at_3_std
      value: 1.5264173782171961
    - type: nauc_precision_at_5_diff1
      value: 23.225740781746087
    - type: nauc_precision_at_5_max
      value: 18.2796460850911
    - type: nauc_precision_at_5_std
      value: 2.1100122710075775
    - type: nauc_recall_at_1000_diff1
      value: 9.817522629457587
    - type: nauc_recall_at_1000_max
      value: 11.67612437506304
    - type: nauc_recall_at_1000_std
      value: 12.86161652413565
    - type: nauc_recall_at_100_diff1
      value: 13.904333022125268
    - type: nauc_recall_at_100_max
      value: 10.499528500128406
    - type: nauc_recall_at_100_std
      value: 4.6754242406439666
    - type: nauc_recall_at_10_diff1
      value: 21.227509281109402
    - type: nauc_recall_at_10_max
      value: 12.138466304037836
    - type: nauc_recall_at_10_std
      value: -1.3590502364993584
    - type: nauc_recall_at_1_diff1
      value: 40.57399794385126
    - type: nauc_recall_at_1_max
      value: 16.149171625208833
    - type: nauc_recall_at_1_std
      value: -2.4218080519307703
    - type: nauc_recall_at_20_diff1
      value: 17.53031727243238
    - type: nauc_recall_at_20_max
      value: 11.315767656629951
    - type: nauc_recall_at_20_std
      value: -0.0815670020293258
    - type: nauc_recall_at_3_diff1
      value: 26.852256743726844
    - type: nauc_recall_at_3_max
      value: 15.854361939492712
    - type: nauc_recall_at_3_std
      value: -1.153367621045346
    - type: nauc_recall_at_5_diff1
      value: 24.348877280666795
    - type: nauc_recall_at_5_max
      value: 14.160377863254098
    - type: nauc_recall_at_5_std
      value: -1.005614215572403
    - type: ndcg_at_1
      value: 9.016
    - type: ndcg_at_10
      value: 13.447000000000001
    - type: ndcg_at_100
      value: 17.307
    - type: ndcg_at_1000
      value: 20.821
    - type: ndcg_at_20
      value: 14.735999999999999
    - type: ndcg_at_3
      value: 11.122
    - type: ndcg_at_5
      value: 12.303
    - type: precision_at_1
      value: 9.016
    - type: precision_at_10
      value: 2.536
    - type: precision_at_100
      value: 0.534
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_20
      value: 1.635
    - type: precision_at_3
      value: 5.391
    - type: precision_at_5
      value: 4.067
    - type: recall_at_1
      value: 7.4990000000000006
    - type: recall_at_10
      value: 18.843
    - type: recall_at_100
      value: 36.508
    - type: recall_at_1000
      value: 62.564
    - type: recall_at_20
      value: 23.538
    - type: recall_at_3
      value: 12.435
    - type: recall_at_5
      value: 15.443999999999999
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackUnixRetrieval (default)
      revision: 6c6430d3a6d36f8d2a829195bc5dc94d7e063e53
      split: test
      type: mteb/cqadupstack-unix
    metrics:
    - type: main_score
      value: 21.052
    - type: map_at_1
      value: 12.426
    - type: map_at_10
      value: 17.587
    - type: map_at_100
      value: 18.442
    - type: map_at_1000
      value: 18.568
    - type: map_at_20
      value: 18.015
    - type: map_at_3
      value: 15.917
    - type: map_at_5
      value: 16.861
    - type: mrr_at_1
      value: 14.645522388059701
    - type: mrr_at_10
      value: 20.31272210376687
    - type: mrr_at_100
      value: 21.125419140238996
    - type: mrr_at_1000
      value: 21.22837687980679
    - type: mrr_at_20
      value: 20.729995038526873
    - type: mrr_at_3
      value: 18.470149253731346
    - type: mrr_at_5
      value: 19.440298507462682
    - type: nauc_map_at_1000_diff1
      value: 39.33163341095681
    - type: nauc_map_at_1000_max
      value: 29.332723485192584
    - type: nauc_map_at_1000_std
      value: -6.31422961090444
    - type: nauc_map_at_100_diff1
      value: 39.356962232089494
    - type: nauc_map_at_100_max
      value: 29.291827447082692
    - type: nauc_map_at_100_std
      value: -6.358110176588159
    - type: nauc_map_at_10_diff1
      value: 39.89765883980389
    - type: nauc_map_at_10_max
      value: 29.498700843025606
    - type: nauc_map_at_10_std
      value: -6.775819565477977
    - type: nauc_map_at_1_diff1
      value: 47.64138101908174
    - type: nauc_map_at_1_max
      value: 36.0817071041123
    - type: nauc_map_at_1_std
      value: -8.59842963175043
    - type: nauc_map_at_20_diff1
      value: 39.55686797301435
    - type: nauc_map_at_20_max
      value: 29.425728630412724
    - type: nauc_map_at_20_std
      value: -6.6498355486245035
    - type: nauc_map_at_3_diff1
      value: 41.55255505626839
    - type: nauc_map_at_3_max
      value: 30.965022540712372
    - type: nauc_map_at_3_std
      value: -7.321893674664093
    - type: nauc_map_at_5_diff1
      value: 40.76985356532814
    - type: nauc_map_at_5_max
      value: 29.511663011170363
    - type: nauc_map_at_5_std
      value: -7.524984450207084
    - type: nauc_mrr_at_1000_diff1
      value: 39.110924778504305
    - type: nauc_mrr_at_1000_max
      value: 30.4866545203556
    - type: nauc_mrr_at_1000_std
      value: -4.889664679047176
    - type: nauc_mrr_at_100_diff1
      value: 39.10714043846902
    - type: nauc_mrr_at_100_max
      value: 30.49597097533955
    - type: nauc_mrr_at_100_std
      value: -4.8975738964109246
    - type: nauc_mrr_at_10_diff1
      value: 39.6637855428504
    - type: nauc_mrr_at_10_max
      value: 30.665144795943938
    - type: nauc_mrr_at_10_std
      value: -5.168074959781463
    - type: nauc_mrr_at_1_diff1
      value: 48.386216593053376
    - type: nauc_mrr_at_1_max
      value: 37.643270167025484
    - type: nauc_mrr_at_1_std
      value: -6.465749697952698
    - type: nauc_mrr_at_20_diff1
      value: 39.27186939293
    - type: nauc_mrr_at_20_max
      value: 30.61566429797341
    - type: nauc_mrr_at_20_std
      value: -5.16582726329683
    - type: nauc_mrr_at_3_diff1
      value: 41.23023142998567
    - type: nauc_mrr_at_3_max
      value: 32.006762101712596
    - type: nauc_mrr_at_3_std
      value: -5.613867880151202
    - type: nauc_mrr_at_5_diff1
      value: 40.471318774809276
    - type: nauc_mrr_at_5_max
      value: 30.358316446904404
    - type: nauc_mrr_at_5_std
      value: -6.042273290129676
    - type: nauc_ndcg_at_1000_diff1
      value: 34.039095525228355
    - type: nauc_ndcg_at_1000_max
      value: 27.339727583523203
    - type: nauc_ndcg_at_1000_std
      value: -2.1385641663234254
    - type: nauc_ndcg_at_100_diff1
      value: 34.50706875780665
    - type: nauc_ndcg_at_100_max
      value: 26.517497925440946
    - type: nauc_ndcg_at_100_std
      value: -3.0287263572679577
    - type: nauc_ndcg_at_10_diff1
      value: 36.68381444467528
    - type: nauc_ndcg_at_10_max
      value: 27.445104277139816
    - type: nauc_ndcg_at_10_std
      value: -5.434347382364577
    - type: nauc_ndcg_at_1_diff1
      value: 48.386216593053376
    - type: nauc_ndcg_at_1_max
      value: 37.643270167025484
    - type: nauc_ndcg_at_1_std
      value: -6.465749697952698
    - type: nauc_ndcg_at_20_diff1
      value: 35.62521976436465
    - type: nauc_ndcg_at_20_max
      value: 27.22284657323637
    - type: nauc_ndcg_at_20_std
      value: -5.212168832453886
    - type: nauc_ndcg_at_3_diff1
      value: 39.65658632289389
    - type: nauc_ndcg_at_3_max
      value: 30.165746048337073
    - type: nauc_ndcg_at_3_std
      value: -6.483184171621599
    - type: nauc_ndcg_at_5_diff1
      value: 38.435574597264896
    - type: nauc_ndcg_at_5_max
      value: 27.29234402819966
    - type: nauc_ndcg_at_5_std
      value: -7.163637857767252
    - type: nauc_precision_at_1000_diff1
      value: -0.5757124881694341
    - type: nauc_precision_at_1000_max
      value: 16.29256700958668
    - type: nauc_precision_at_1000_std
      value: 12.307495698059249
    - type: nauc_precision_at_100_diff1
      value: 14.066163778136765
    - type: nauc_precision_at_100_max
      value: 19.90375717448338
    - type: nauc_precision_at_100_std
      value: 8.429418360577989
    - type: nauc_precision_at_10_diff1
      value: 27.40062141138924
    - type: nauc_precision_at_10_max
      value: 24.331893027882526
    - type: nauc_precision_at_10_std
      value: -0.14920457766396567
    - type: nauc_precision_at_1_diff1
      value: 48.386216593053376
    - type: nauc_precision_at_1_max
      value: 37.643270167025484
    - type: nauc_precision_at_1_std
      value: -6.465749697952698
    - type: nauc_precision_at_20_diff1
      value: 22.37358263091247
    - type: nauc_precision_at_20_max
      value: 22.520668242414693
    - type: nauc_precision_at_20_std
      value: -0.08668534976010908
    - type: nauc_precision_at_3_diff1
      value: 34.96831803301659
    - type: nauc_precision_at_3_max
      value: 26.776959232155157
    - type: nauc_precision_at_3_std
      value: -4.066190254588547
    - type: nauc_precision_at_5_diff1
      value: 31.927749511846642
    - type: nauc_precision_at_5_max
      value: 22.133595472823778
    - type: nauc_precision_at_5_std
      value: -4.659887568906566
    - type: nauc_recall_at_1000_diff1
      value: 9.864518875286782
    - type: nauc_recall_at_1000_max
      value: 16.079783011341842
    - type: nauc_recall_at_1000_std
      value: 18.422744210796466
    - type: nauc_recall_at_100_diff1
      value: 19.825105290912774
    - type: nauc_recall_at_100_max
      value: 15.376507283040288
    - type: nauc_recall_at_100_std
      value: 6.656479057915567
    - type: nauc_recall_at_10_diff1
      value: 27.968188448369634
    - type: nauc_recall_at_10_max
      value: 20.468242077216573
    - type: nauc_recall_at_10_std
      value: -3.195299662368557
    - type: nauc_recall_at_1_diff1
      value: 47.64138101908174
    - type: nauc_recall_at_1_max
      value: 36.0817071041123
    - type: nauc_recall_at_1_std
      value: -8.59842963175043
    - type: nauc_recall_at_20_diff1
      value: 24.881472147156558
    - type: nauc_recall_at_20_max
      value: 19.539649229845484
    - type: nauc_recall_at_20_std
      value: -2.597679057149296
    - type: nauc_recall_at_3_diff1
      value: 35.011852772709275
    - type: nauc_recall_at_3_max
      value: 25.42121427465917
    - type: nauc_recall_at_3_std
      value: -6.475117977119084
    - type: nauc_recall_at_5_diff1
      value: 32.33891575382334
    - type: nauc_recall_at_5_max
      value: 19.84609317314949
    - type: nauc_recall_at_5_std
      value: -7.713982073904097
    - type: ndcg_at_1
      value: 14.646
    - type: ndcg_at_10
      value: 21.052
    - type: ndcg_at_100
      value: 25.503999999999998
    - type: ndcg_at_1000
      value: 28.98
    - type: ndcg_at_20
      value: 22.595000000000002
    - type: ndcg_at_3
      value: 17.736
    - type: ndcg_at_5
      value: 19.283
    - type: precision_at_1
      value: 14.646
    - type: precision_at_10
      value: 3.703
    - type: precision_at_100
      value: 0.658
    - type: precision_at_1000
      value: 0.107
    - type: precision_at_20
      value: 2.248
    - type: precision_at_3
      value: 8.209
    - type: precision_at_5
      value: 5.933
    - type: recall_at_1
      value: 12.426
    - type: recall_at_10
      value: 28.977999999999998
    - type: recall_at_100
      value: 49.309
    - type: recall_at_1000
      value: 74.90599999999999
    - type: recall_at_20
      value: 34.777
    - type: recall_at_3
      value: 19.975
    - type: recall_at_5
      value: 23.848
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackWebmastersRetrieval (default)
      revision: 160c094312a0e1facb97e55eeddb698c0abe3571
      split: test
      type: mteb/cqadupstack-webmasters
    metrics:
    - type: main_score
      value: 24.541
    - type: map_at_1
      value: 13.947999999999999
    - type: map_at_10
      value: 20.217
    - type: map_at_100
      value: 21.542
    - type: map_at_1000
      value: 21.731
    - type: map_at_20
      value: 20.918
    - type: map_at_3
      value: 18.422
    - type: map_at_5
      value: 19.308
    - type: mrr_at_1
      value: 17.588932806324113
    - type: mrr_at_10
      value: 23.858695652173907
    - type: mrr_at_100
      value: 24.851399845780602
    - type: mrr_at_1000
      value: 24.935814602119343
    - type: mrr_at_20
      value: 24.408561689686646
    - type: mrr_at_3
      value: 22.068511198945977
    - type: mrr_at_5
      value: 22.947957839262187
    - type: nauc_map_at_1000_diff1
      value: 43.33859805395678
    - type: nauc_map_at_1000_max
      value: 16.794319215595962
    - type: nauc_map_at_1000_std
      value: 1.1069609672810294
    - type: nauc_map_at_100_diff1
      value: 43.31953768477659
    - type: nauc_map_at_100_max
      value: 16.80649217527658
    - type: nauc_map_at_100_std
      value: 0.9267847714960188
    - type: nauc_map_at_10_diff1
      value: 44.20016434158565
    - type: nauc_map_at_10_max
      value: 17.616372385010706
    - type: nauc_map_at_10_std
      value: 0.9811655928155236
    - type: nauc_map_at_1_diff1
      value: 54.077090204227886
    - type: nauc_map_at_1_max
      value: 19.709908044012398
    - type: nauc_map_at_1_std
      value: -3.6121421783838605
    - type: nauc_map_at_20_diff1
      value: 43.562495604434936
    - type: nauc_map_at_20_max
      value: 16.862716386830666
    - type: nauc_map_at_20_std
      value: 0.5191844288914963
    - type: nauc_map_at_3_diff1
      value: 45.711284130752844
    - type: nauc_map_at_3_max
      value: 17.36946450306621
    - type: nauc_map_at_3_std
      value: -0.7230194301728452
    - type: nauc_map_at_5_diff1
      value: 44.75072357358313
    - type: nauc_map_at_5_max
      value: 17.32559857940083
    - type: nauc_map_at_5_std
      value: 0.8658883165628524
    - type: nauc_mrr_at_1000_diff1
      value: 43.02583464427709
    - type: nauc_mrr_at_1000_max
      value: 17.219554017696893
    - type: nauc_mrr_at_1000_std
      value: 1.7672153931182881
    - type: nauc_mrr_at_100_diff1
      value: 42.99189791126074
    - type: nauc_mrr_at_100_max
      value: 17.192364035400853
    - type: nauc_mrr_at_100_std
      value: 1.756535126973775
    - type: nauc_mrr_at_10_diff1
      value: 43.08444131617144
    - type: nauc_mrr_at_10_max
      value: 17.29015910503535
    - type: nauc_mrr_at_10_std
      value: 1.7125365374735448
    - type: nauc_mrr_at_1_diff1
      value: 50.5556261668236
    - type: nauc_mrr_at_1_max
      value: 19.294675141150243
    - type: nauc_mrr_at_1_std
      value: -2.3442346201176356
    - type: nauc_mrr_at_20_diff1
      value: 42.88013095653635
    - type: nauc_mrr_at_20_max
      value: 17.00639653815113
    - type: nauc_mrr_at_20_std
      value: 1.5053330754970131
    - type: nauc_mrr_at_3_diff1
      value: 44.598903878474786
    - type: nauc_mrr_at_3_max
      value: 17.615736504224426
    - type: nauc_mrr_at_3_std
      value: 0.2407821472745429
    - type: nauc_mrr_at_5_diff1
      value: 43.903767716493434
    - type: nauc_mrr_at_5_max
      value: 17.209076005917304
    - type: nauc_mrr_at_5_std
      value: 1.7097976325552382
    - type: nauc_ndcg_at_1000_diff1
      value: 40.41806553114964
    - type: nauc_ndcg_at_1000_max
      value: 17.795144923496352
    - type: nauc_ndcg_at_1000_std
      value: 6.107826233525735
    - type: nauc_ndcg_at_100_diff1
      value: 39.13804206519448
    - type: nauc_ndcg_at_100_max
      value: 16.705953947028362
    - type: nauc_ndcg_at_100_std
      value: 5.151889715872744
    - type: nauc_ndcg_at_10_diff1
      value: 40.77046801594463
    - type: nauc_ndcg_at_10_max
      value: 17.398314688629597
    - type: nauc_ndcg_at_10_std
      value: 3.6749729455778537
    - type: nauc_ndcg_at_1_diff1
      value: 50.5556261668236
    - type: nauc_ndcg_at_1_max
      value: 19.294675141150243
    - type: nauc_ndcg_at_1_std
      value: -2.3442346201176356
    - type: nauc_ndcg_at_20_diff1
      value: 39.48069335422461
    - type: nauc_ndcg_at_20_max
      value: 15.585590458050417
    - type: nauc_ndcg_at_20_std
      value: 2.6822276938545344
    - type: nauc_ndcg_at_3_diff1
      value: 42.229898693167826
    - type: nauc_ndcg_at_3_max
      value: 16.613922904445026
    - type: nauc_ndcg_at_3_std
      value: 1.3855074174214905
    - type: nauc_ndcg_at_5_diff1
      value: 41.60713305096391
    - type: nauc_ndcg_at_5_max
      value: 16.484928388833886
    - type: nauc_ndcg_at_5_std
      value: 3.4073283752555845
    - type: nauc_precision_at_1000_diff1
      value: 0.3356504625232749
    - type: nauc_precision_at_1000_max
      value: 4.49983560064616
    - type: nauc_precision_at_1000_std
      value: 14.303080258454347
    - type: nauc_precision_at_100_diff1
      value: 8.835404291542027
    - type: nauc_precision_at_100_max
      value: 3.150943200684829
    - type: nauc_precision_at_100_std
      value: 15.048767403827714
    - type: nauc_precision_at_10_diff1
      value: 21.52838092138577
    - type: nauc_precision_at_10_max
      value: 12.571532848032687
    - type: nauc_precision_at_10_std
      value: 8.128913556180294
    - type: nauc_precision_at_1_diff1
      value: 50.5556261668236
    - type: nauc_precision_at_1_max
      value: 19.294675141150243
    - type: nauc_precision_at_1_std
      value: -2.3442346201176356
    - type: nauc_precision_at_20_diff1
      value: 14.781087100941523
    - type: nauc_precision_at_20_max
      value: 5.988342267683038
    - type: nauc_precision_at_20_std
      value: 6.082560933226635
    - type: nauc_precision_at_3_diff1
      value: 31.392623045190426
    - type: nauc_precision_at_3_max
      value: 14.718033879054303
    - type: nauc_precision_at_3_std
      value: 4.660197047853719
    - type: nauc_precision_at_5_diff1
      value: 27.092109888977834
    - type: nauc_precision_at_5_max
      value: 13.203296668102244
    - type: nauc_precision_at_5_std
      value: 7.456529374803908
    - type: nauc_recall_at_1000_diff1
      value: 27.98171442888675
    - type: nauc_recall_at_1000_max
      value: 25.506878355571327
    - type: nauc_recall_at_1000_std
      value: 36.45225778705996
    - type: nauc_recall_at_100_diff1
      value: 24.89136039905334
    - type: nauc_recall_at_100_max
      value: 15.57355424120746
    - type: nauc_recall_at_100_std
      value: 18.274980258425174
    - type: nauc_recall_at_10_diff1
      value: 32.680187174796224
    - type: nauc_recall_at_10_max
      value: 17.80036586473841
    - type: nauc_recall_at_10_std
      value: 9.172008959628027
    - type: nauc_recall_at_1_diff1
      value: 54.077090204227886
    - type: nauc_recall_at_1_max
      value: 19.709908044012398
    - type: nauc_recall_at_1_std
      value: -3.6121421783838605
    - type: nauc_recall_at_20_diff1
      value: 28.310306054990818
    - type: nauc_recall_at_20_max
      value: 11.540330817577258
    - type: nauc_recall_at_20_std
      value: 5.916827349873026
    - type: nauc_recall_at_3_diff1
      value: 37.31685917439632
    - type: nauc_recall_at_3_max
      value: 15.303688489057219
    - type: nauc_recall_at_3_std
      value: 3.1192461784588934
    - type: nauc_recall_at_5_diff1
      value: 34.98096682397507
    - type: nauc_recall_at_5_max
      value: 14.370025231136207
    - type: nauc_recall_at_5_std
      value: 8.02340098284342
    - type: ndcg_at_1
      value: 17.589
    - type: ndcg_at_10
      value: 24.541
    - type: ndcg_at_100
      value: 30.098999999999997
    - type: ndcg_at_1000
      value: 33.522999999999996
    - type: ndcg_at_20
      value: 26.608999999999998
    - type: ndcg_at_3
      value: 21.587
    - type: ndcg_at_5
      value: 22.726
    - type: precision_at_1
      value: 17.589
    - type: precision_at_10
      value: 4.862
    - type: precision_at_100
      value: 1.109
    - type: precision_at_1000
      value: 0.202
    - type: precision_at_20
      value: 3.241
    - type: precision_at_3
      value: 10.54
    - type: precision_at_5
      value: 7.549
    - type: recall_at_1
      value: 13.947999999999999
    - type: recall_at_10
      value: 32.962
    - type: recall_at_100
      value: 58.475
    - type: recall_at_1000
      value: 81.281
    - type: recall_at_20
      value: 40.963
    - type: recall_at_3
      value: 23.654
    - type: recall_at_5
      value: 26.976
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB CQADupstackWordpressRetrieval (default)
      revision: 4ffe81d471b1924886b33c7567bfb200e9eec5c4
      split: test
      type: mteb/cqadupstack-wordpress
    metrics:
    - type: main_score
      value: 18.494
    - type: map_at_1
      value: 12.113
    - type: map_at_10
      value: 15.742
    - type: map_at_100
      value: 16.588
    - type: map_at_1000
      value: 16.695999999999998
    - type: map_at_20
      value: 16.133
    - type: map_at_3
      value: 14.235999999999999
    - type: map_at_5
      value: 14.863999999999999
    - type: mrr_at_1
      value: 13.123844731977819
    - type: mrr_at_10
      value: 17.048748643018513
    - type: mrr_at_100
      value: 17.92317861781634
    - type: mrr_at_1000
      value: 18.01677808515946
    - type: mrr_at_20
      value: 17.49017709474638
    - type: mrr_at_3
      value: 15.372766481823783
    - type: mrr_at_5
      value: 16.09365372766482
    - type: nauc_map_at_1000_diff1
      value: 29.177576432015336
    - type: nauc_map_at_1000_max
      value: 15.675654084454017
    - type: nauc_map_at_1000_std
      value: -4.940369928551305
    - type: nauc_map_at_100_diff1
      value: 29.17418032816123
    - type: nauc_map_at_100_max
      value: 15.70613463825502
    - type: nauc_map_at_100_std
      value: -4.980070415403916
    - type: nauc_map_at_10_diff1
      value: 29.52509707922253
    - type: nauc_map_at_10_max
      value: 15.554777819804485
    - type: nauc_map_at_10_std
      value: -5.659659507621501
    - type: nauc_map_at_1_diff1
      value: 34.19820381105881
    - type: nauc_map_at_1_max
      value: 19.40226008766731
    - type: nauc_map_at_1_std
      value: -6.841796961195136
    - type: nauc_map_at_20_diff1
      value: 29.25644111895755
    - type: nauc_map_at_20_max
      value: 15.60193203494247
    - type: nauc_map_at_20_std
      value: -5.234632550119407
    - type: nauc_map_at_3_diff1
      value: 29.839926969290953
    - type: nauc_map_at_3_max
      value: 15.167659779156203
    - type: nauc_map_at_3_std
      value: -7.424881973369861
    - type: nauc_map_at_5_diff1
      value: 29.554402844383716
    - type: nauc_map_at_5_max
      value: 15.476818672588767
    - type: nauc_map_at_5_std
      value: -6.489967221653721
    - type: nauc_mrr_at_1000_diff1
      value: 28.705818533659276
    - type: nauc_mrr_at_1000_max
      value: 16.67009666276028
    - type: nauc_mrr_at_1000_std
      value: -4.8854799116351115
    - type: nauc_mrr_at_100_diff1
      value: 28.686498093818564
    - type: nauc_mrr_at_100_max
      value: 16.688173945854402
    - type: nauc_mrr_at_100_std
      value: -4.90598798803822
    - type: nauc_mrr_at_10_diff1
      value: 28.82460941424038
    - type: nauc_mrr_at_10_max
      value: 16.510644042417084
    - type: nauc_mrr_at_10_std
      value: -5.339958505172009
    - type: nauc_mrr_at_1_diff1
      value: 34.33807774677732
    - type: nauc_mrr_at_1_max
      value: 20.45291712510333
    - type: nauc_mrr_at_1_std
      value: -6.922645236892996
    - type: nauc_mrr_at_20_diff1
      value: 28.69247665330908
    - type: nauc_mrr_at_20_max
      value: 16.62791528528386
    - type: nauc_mrr_at_20_std
      value: -5.092785887600711
    - type: nauc_mrr_at_3_diff1
      value: 29.290860767239312
    - type: nauc_mrr_at_3_max
      value: 16.38255487970124
    - type: nauc_mrr_at_3_std
      value: -7.432562336997352
    - type: nauc_mrr_at_5_diff1
      value: 28.828148893662398
    - type: nauc_mrr_at_5_max
      value: 16.912586765582905
    - type: nauc_mrr_at_5_std
      value: -6.185804675986914
    - type: nauc_ndcg_at_1000_diff1
      value: 27.309410657571785
    - type: nauc_ndcg_at_1000_max
      value: 14.649682696366165
    - type: nauc_ndcg_at_1000_std
      value: -0.5489983812758055
    - type: nauc_ndcg_at_100_diff1
      value: 26.910525741498393
    - type: nauc_ndcg_at_100_max
      value: 15.076985446932193
    - type: nauc_ndcg_at_100_std
      value: -1.8950871581487898
    - type: nauc_ndcg_at_10_diff1
      value: 28.00932597205862
    - type: nauc_ndcg_at_10_max
      value: 14.797077867874442
    - type: nauc_ndcg_at_10_std
      value: -4.267843055074893
    - type: nauc_ndcg_at_1_diff1
      value: 34.33807774677732
    - type: nauc_ndcg_at_1_max
      value: 20.45291712510333
    - type: nauc_ndcg_at_1_std
      value: -6.922645236892996
    - type: nauc_ndcg_at_20_diff1
      value: 27.382899464152366
    - type: nauc_ndcg_at_20_max
      value: 14.986274451755072
    - type: nauc_ndcg_at_20_std
      value: -3.0822069038645665
    - type: nauc_ndcg_at_3_diff1
      value: 28.432220536421777
    - type: nauc_ndcg_at_3_max
      value: 14.321785354708192
    - type: nauc_ndcg_at_3_std
      value: -7.499686293742728
    - type: nauc_ndcg_at_5_diff1
      value: 27.880753061285805
    - type: nauc_ndcg_at_5_max
      value: 14.872276671266299
    - type: nauc_ndcg_at_5_std
      value: -5.843553028670821
    - type: nauc_precision_at_1000_diff1
      value: 1.2054794765973806
    - type: nauc_precision_at_1000_max
      value: -2.9890375928858197
    - type: nauc_precision_at_1000_std
      value: 9.673411560271315
    - type: nauc_precision_at_100_diff1
      value: 15.520027853569504
    - type: nauc_precision_at_100_max
      value: 10.613328032221544
    - type: nauc_precision_at_100_std
      value: 6.202043473690328
    - type: nauc_precision_at_10_diff1
      value: 22.519305955225967
    - type: nauc_precision_at_10_max
      value: 13.175610226423181
    - type: nauc_precision_at_10_std
      value: 0.20653933581209644
    - type: nauc_precision_at_1_diff1
      value: 34.33807774677732
    - type: nauc_precision_at_1_max
      value: 20.45291712510333
    - type: nauc_precision_at_1_std
      value: -6.922645236892996
    - type: nauc_precision_at_20_diff1
      value: 20.245876915517066
    - type: nauc_precision_at_20_max
      value: 13.163241999366477
    - type: nauc_precision_at_20_std
      value: 3.31158401781611
    - type: nauc_precision_at_3_diff1
      value: 23.83680559850096
    - type: nauc_precision_at_3_max
      value: 11.393759154805142
    - type: nauc_precision_at_3_std
      value: -8.438766003819104
    - type: nauc_precision_at_5_diff1
      value: 22.838234962034168
    - type: nauc_precision_at_5_max
      value: 13.881058833377649
    - type: nauc_precision_at_5_std
      value: -3.7337045513771745
    - type: nauc_recall_at_1000_diff1
      value: 22.15205340453375
    - type: nauc_recall_at_1000_max
      value: 8.73027409332576
    - type: nauc_recall_at_1000_std
      value: 22.37256072062562
    - type: nauc_recall_at_100_diff1
      value: 20.88031166008365
    - type: nauc_recall_at_100_max
      value: 12.338348522414508
    - type: nauc_recall_at_100_std
      value: 6.332498952277319
    - type: nauc_recall_at_10_diff1
      value: 25.400373762163714
    - type: nauc_recall_at_10_max
      value: 12.475974675193605
    - type: nauc_recall_at_10_std
      value: -1.2784524840082803
    - type: nauc_recall_at_1_diff1
      value: 34.19820381105881
    - type: nauc_recall_at_1_max
      value: 19.40226008766731
    - type: nauc_recall_at_1_std
      value: -6.841796961195136
    - type: nauc_recall_at_20_diff1
      value: 23.65244983531732
    - type: nauc_recall_at_20_max
      value: 12.939746629309528
    - type: nauc_recall_at_20_std
      value: 1.9368049837461982
    - type: nauc_recall_at_3_diff1
      value: 25.239472329225578
    - type: nauc_recall_at_3_max
      value: 10.851916276600768
    - type: nauc_recall_at_3_std
      value: -7.845478941175809
    - type: nauc_recall_at_5_diff1
      value: 24.587232792243803
    - type: nauc_recall_at_5_max
      value: 12.327381094603336
    - type: nauc_recall_at_5_std
      value: -4.431325781211059
    - type: ndcg_at_1
      value: 13.123999999999999
    - type: ndcg_at_10
      value: 18.494
    - type: ndcg_at_100
      value: 23.307
    - type: ndcg_at_1000
      value: 26.522000000000002
    - type: ndcg_at_20
      value: 19.932
    - type: ndcg_at_3
      value: 15.226999999999999
    - type: ndcg_at_5
      value: 16.352
    - type: precision_at_1
      value: 13.123999999999999
    - type: precision_at_10
      value: 2.994
    - type: precision_at_100
      value: 0.588
    - type: precision_at_1000
      value: 0.093
    - type: precision_at_20
      value: 1.848
    - type: precision_at_3
      value: 6.161
    - type: precision_at_5
      value: 4.324999999999999
    - type: recall_at_1
      value: 12.113
    - type: recall_at_10
      value: 25.912000000000003
    - type: recall_at_100
      value: 49.112
    - type: recall_at_1000
      value: 74.208
    - type: recall_at_20
      value: 31.226
    - type: recall_at_3
      value: 16.956
    - type: recall_at_5
      value: 19.667
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB ClimateFEVER (default)
      revision: 47f2ac6acb640fc46020b02a5b59fdda04d39380
      split: test
      type: mteb/climate-fever
    metrics:
    - type: main_score
      value: 19.737
    - type: map_at_1
      value: 7.580000000000001
    - type: map_at_10
      value: 13.288
    - type: map_at_100
      value: 14.777999999999999
    - type: map_at_1000
      value: 14.982000000000001
    - type: map_at_20
      value: 13.969000000000001
    - type: map_at_3
      value: 10.871
    - type: map_at_5
      value: 12.009
    - type: mrr_at_1
      value: 17.133550488599347
    - type: mrr_at_10
      value: 26.558450959102398
    - type: mrr_at_100
      value: 27.706190405221943
    - type: mrr_at_1000
      value: 27.769512197225342
    - type: mrr_at_20
      value: 27.2873997190966
    - type: mrr_at_3
      value: 23.203040173724187
    - type: mrr_at_5
      value: 24.97176981541797
    - type: nauc_map_at_1000_diff1
      value: 24.72960374178977
    - type: nauc_map_at_1000_max
      value: 24.480735633901148
    - type: nauc_map_at_1000_std
      value: 13.793019923391537
    - type: nauc_map_at_100_diff1
      value: 24.70242778622659
    - type: nauc_map_at_100_max
      value: 24.312935578677994
    - type: nauc_map_at_100_std
      value: 13.557776976404154
    - type: nauc_map_at_10_diff1
      value: 25.062343539223704
    - type: nauc_map_at_10_max
      value: 23.281714659693968
    - type: nauc_map_at_10_std
      value: 11.764008031776111
    - type: nauc_map_at_1_diff1
      value: 34.84443951215321
    - type: nauc_map_at_1_max
      value: 23.27803526667906
    - type: nauc_map_at_1_std
      value: 6.533433240791793
    - type: nauc_map_at_20_diff1
      value: 24.829658985975623
    - type: nauc_map_at_20_max
      value: 23.479388707533953
    - type: nauc_map_at_20_std
      value: 12.527441286050017
    - type: nauc_map_at_3_diff1
      value: 25.332429880275175
    - type: nauc_map_at_3_max
      value: 20.852134738989523
    - type: nauc_map_at_3_std
      value: 8.451908175536312
    - type: nauc_map_at_5_diff1
      value: 24.700749559954794
    - type: nauc_map_at_5_max
      value: 21.513612585855956
    - type: nauc_map_at_5_std
      value: 9.878132842170151
    - type: nauc_mrr_at_1000_diff1
      value: 21.427128632298302
    - type: nauc_mrr_at_1000_max
      value: 26.992365852241146
    - type: nauc_mrr_at_1000_std
      value: 17.321690450890923
    - type: nauc_mrr_at_100_diff1
      value: 21.427383085782928
    - type: nauc_mrr_at_100_max
      value: 26.98184853141775
    - type: nauc_mrr_at_100_std
      value: 17.328094756874613
    - type: nauc_mrr_at_10_diff1
      value: 21.369108071516347
    - type: nauc_mrr_at_10_max
      value: 26.902002317172485
    - type: nauc_mrr_at_10_std
      value: 16.94045381313827
    - type: nauc_mrr_at_1_diff1
      value: 28.59157297539116
    - type: nauc_mrr_at_1_max
      value: 26.541603377090272
    - type: nauc_mrr_at_1_std
      value: 12.319135936486488
    - type: nauc_mrr_at_20_diff1
      value: 21.2637422505419
    - type: nauc_mrr_at_20_max
      value: 26.891762697769135
    - type: nauc_mrr_at_20_std
      value: 17.332417370892916
    - type: nauc_mrr_at_3_diff1
      value: 21.531125666898667
    - type: nauc_mrr_at_3_max
      value: 25.0422060271623
    - type: nauc_mrr_at_3_std
      value: 14.347699109361416
    - type: nauc_mrr_at_5_diff1
      value: 20.90283927121716
    - type: nauc_mrr_at_5_max
      value: 25.881301712313892
    - type: nauc_mrr_at_5_std
      value: 15.7460269053894
    - type: nauc_ndcg_at_1000_diff1
      value: 21.30311098816425
    - type: nauc_ndcg_at_1000_max
      value: 30.867798726301167
    - type: nauc_ndcg_at_1000_std
      value: 24.11583525958116
    - type: nauc_ndcg_at_100_diff1
      value: 20.90637661138601
    - type: nauc_ndcg_at_100_max
      value: 28.903383155219046
    - type: nauc_ndcg_at_100_std
      value: 21.700698920603987
    - type: nauc_ndcg_at_10_diff1
      value: 21.877322734445865
    - type: nauc_ndcg_at_10_max
      value: 25.722202242090937
    - type: nauc_ndcg_at_10_std
      value: 16.313273898123086
    - type: nauc_ndcg_at_1_diff1
      value: 28.59157297539116
    - type: nauc_ndcg_at_1_max
      value: 26.541603377090272
    - type: nauc_ndcg_at_1_std
      value: 12.319135936486488
    - type: nauc_ndcg_at_20_diff1
      value: 21.352023081638986
    - type: nauc_ndcg_at_20_max
      value: 26.023607863730085
    - type: nauc_ndcg_at_20_std
      value: 18.159966552384855
    - type: nauc_ndcg_at_3_diff1
      value: 21.584694082829326
    - type: nauc_ndcg_at_3_max
      value: 22.277158070601683
    - type: nauc_ndcg_at_3_std
      value: 11.280093739110814
    - type: nauc_ndcg_at_5_diff1
      value: 21.22794762775943
    - type: nauc_ndcg_at_5_max
      value: 22.814546433767998
    - type: nauc_ndcg_at_5_std
      value: 13.07693075742485
    - type: nauc_precision_at_1000_diff1
      value: 1.229127826394365
    - type: nauc_precision_at_1000_max
      value: 30.428220237747745
    - type: nauc_precision_at_1000_std
      value: 35.48684383703058
    - type: nauc_precision_at_100_diff1
      value: 5.839822502358944
    - type: nauc_precision_at_100_max
      value: 31.749689157158727
    - type: nauc_precision_at_100_std
      value: 33.030884537557846
    - type: nauc_precision_at_10_diff1
      value: 13.618894319397986
    - type: nauc_precision_at_10_max
      value: 29.944256877239983
    - type: nauc_precision_at_10_std
      value: 25.16799740254062
    - type: nauc_precision_at_1_diff1
      value: 28.59157297539116
    - type: nauc_precision_at_1_max
      value: 26.541603377090272
    - type: nauc_precision_at_1_std
      value: 12.319135936486488
    - type: nauc_precision_at_20_diff1
      value: 11.016546432068667
    - type: nauc_precision_at_20_max
      value: 29.384029988434037
    - type: nauc_precision_at_20_std
      value: 28.29412908144535
    - type: nauc_precision_at_3_diff1
      value: 14.486980680974531
    - type: nauc_precision_at_3_max
      value: 22.941300149012807
    - type: nauc_precision_at_3_std
      value: 15.948074376558303
    - type: nauc_precision_at_5_diff1
      value: 13.059031709771709
    - type: nauc_precision_at_5_max
      value: 25.538800002473216
    - type: nauc_precision_at_5_std
      value: 20.292315419905833
    - type: nauc_recall_at_1000_diff1
      value: 12.480839027412872
    - type: nauc_recall_at_1000_max
      value: 37.93110145252946
    - type: nauc_recall_at_1000_std
      value: 42.06127213739258
    - type: nauc_recall_at_100_diff1
      value: 11.493155880852662
    - type: nauc_recall_at_100_max
      value: 27.360977043238695
    - type: nauc_recall_at_100_std
      value: 28.13450624683099
    - type: nauc_recall_at_10_diff1
      value: 16.27123527594846
    - type: nauc_recall_at_10_max
      value: 23.449040506685094
    - type: nauc_recall_at_10_std
      value: 17.88389362914695
    - type: nauc_recall_at_1_diff1
      value: 34.84443951215321
    - type: nauc_recall_at_1_max
      value: 23.27803526667906
    - type: nauc_recall_at_1_std
      value: 6.533433240791793
    - type: nauc_recall_at_20_diff1
      value: 14.084418800297216
    - type: nauc_recall_at_20_max
      value: 21.81909793663393
    - type: nauc_recall_at_20_std
      value: 20.44410671487353
    - type: nauc_recall_at_3_diff1
      value: 17.842240049745378
    - type: nauc_recall_at_3_max
      value: 18.51009355096147
    - type: nauc_recall_at_3_std
      value: 9.601196022949464
    - type: nauc_recall_at_5_diff1
      value: 15.974583965720079
    - type: nauc_recall_at_5_max
      value: 19.18647734660961
    - type: nauc_recall_at_5_std
      value: 12.592886956245533
    - type: ndcg_at_1
      value: 17.134
    - type: ndcg_at_10
      value: 19.737
    - type: ndcg_at_100
      value: 26.605
    - type: ndcg_at_1000
      value: 30.625999999999998
    - type: ndcg_at_20
      value: 22.049
    - type: ndcg_at_3
      value: 15.219
    - type: ndcg_at_5
      value: 16.730999999999998
    - type: precision_at_1
      value: 17.134
    - type: precision_at_10
      value: 6.502
    - type: precision_at_100
      value: 1.393
    - type: precision_at_1000
      value: 0.213
    - type: precision_at_20
      value: 4.215
    - type: precision_at_3
      value: 11.488
    - type: precision_at_5
      value: 9.121
    - type: recall_at_1
      value: 7.580000000000001
    - type: recall_at_10
      value: 24.907
    - type: recall_at_100
      value: 49.186
    - type: recall_at_1000
      value: 72.18299999999999
    - type: recall_at_20
      value: 31.623
    - type: recall_at_3
      value: 14.111
    - type: recall_at_5
      value: 18.141
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB DBPedia (default)
      revision: c0f706b76e590d620bd6618b3ca8efdd34e2d659
      split: test
      type: mteb/dbpedia
    metrics:
    - type: main_score
      value: 25.516
    - type: map_at_1
      value: 4.585
    - type: map_at_10
      value: 10.558
    - type: map_at_100
      value: 14.885000000000002
    - type: map_at_1000
      value: 15.915000000000001
    - type: map_at_20
      value: 12.242
    - type: map_at_3
      value: 7.393
    - type: map_at_5
      value: 8.594
    - type: mrr_at_1
      value: 44.5
    - type: mrr_at_10
      value: 54.874107142857135
    - type: mrr_at_100
      value: 55.48600960098919
    - type: mrr_at_1000
      value: 55.51079614224205
    - type: mrr_at_20
      value: 55.27255238526521
    - type: mrr_at_3
      value: 52.04166666666667
    - type: mrr_at_5
      value: 53.516666666666666
    - type: nauc_map_at_1000_diff1
      value: 28.409792363817722
    - type: nauc_map_at_1000_max
      value: 29.789652299261782
    - type: nauc_map_at_1000_std
      value: 30.88764977969059
    - type: nauc_map_at_100_diff1
      value: 28.73278145547276
    - type: nauc_map_at_100_max
      value: 27.28526713063023
    - type: nauc_map_at_100_std
      value: 28.033272815666855
    - type: nauc_map_at_10_diff1
      value: 35.33410925649162
    - type: nauc_map_at_10_max
      value: 11.181399501220257
    - type: nauc_map_at_10_std
      value: 13.180775434092862
    - type: nauc_map_at_1_diff1
      value: 45.86732346705817
    - type: nauc_map_at_1_max
      value: -0.7337028924656861
    - type: nauc_map_at_1_std
      value: 3.969007656387463
    - type: nauc_map_at_20_diff1
      value: 32.737563371691465
    - type: nauc_map_at_20_max
      value: 16.712015714292203
    - type: nauc_map_at_20_std
      value: 18.179837825281563
    - type: nauc_map_at_3_diff1
      value: 39.35894559294948
    - type: nauc_map_at_3_max
      value: 5.448695676931316
    - type: nauc_map_at_3_std
      value: 6.204172744817759
    - type: nauc_map_at_5_diff1
      value: 37.92885448467568
    - type: nauc_map_at_5_max
      value: 6.596337276392895
    - type: nauc_map_at_5_std
      value: 8.235596939594137
    - type: nauc_mrr_at_1000_diff1
      value: 33.42073777564439
    - type: nauc_mrr_at_1000_max
      value: 46.136596890287926
    - type: nauc_mrr_at_1000_std
      value: 29.278108581352296
    - type: nauc_mrr_at_100_diff1
      value: 33.43760365632699
    - type: nauc_mrr_at_100_max
      value: 46.15503090492029
    - type: nauc_mrr_at_100_std
      value: 29.291839612032778
    - type: nauc_mrr_at_10_diff1
      value: 33.37294072436522
    - type: nauc_mrr_at_10_max
      value: 45.88502015727753
    - type: nauc_mrr_at_10_std
      value: 28.974501161757132
    - type: nauc_mrr_at_1_diff1
      value: 37.038949494421324
    - type: nauc_mrr_at_1_max
      value: 43.6843515716405
    - type: nauc_mrr_at_1_std
      value: 29.062767577601583
    - type: nauc_mrr_at_20_diff1
      value: 33.44943708493421
    - type: nauc_mrr_at_20_max
      value: 46.100969958613554
    - type: nauc_mrr_at_20_std
      value: 29.137551697063817
    - type: nauc_mrr_at_3_diff1
      value: 32.932379921286085
    - type: nauc_mrr_at_3_max
      value: 45.830465307372144
    - type: nauc_mrr_at_3_std
      value: 28.98038995101691
    - type: nauc_mrr_at_5_diff1
      value: 33.01848752761743
    - type: nauc_mrr_at_5_max
      value: 45.74337639822611
    - type: nauc_mrr_at_5_std
      value: 28.57219445985183
    - type: nauc_ndcg_at_1000_diff1
      value: 29.031510049028775
    - type: nauc_ndcg_at_1000_max
      value: 40.5267703197412
    - type: nauc_ndcg_at_1000_std
      value: 42.3005865676892
    - type: nauc_ndcg_at_100_diff1
      value: 29.344739057831067
    - type: nauc_ndcg_at_100_max
      value: 33.37828801741486
    - type: nauc_ndcg_at_100_std
      value: 33.98191361448277
    - type: nauc_ndcg_at_10_diff1
      value: 31.589394876613508
    - type: nauc_ndcg_at_10_max
      value: 33.05362616825694
    - type: nauc_ndcg_at_10_std
      value: 28.519650410818052
    - type: nauc_ndcg_at_1_diff1
      value: 35.29470063230581
    - type: nauc_ndcg_at_1_max
      value: 31.551140366841896
    - type: nauc_ndcg_at_1_std
      value: 21.389198724937096
    - type: nauc_ndcg_at_20_diff1
      value: 31.43160990986207
    - type: nauc_ndcg_at_20_max
      value: 30.06950946963706
    - type: nauc_ndcg_at_20_std
      value: 27.355004276047907
    - type: nauc_ndcg_at_3_diff1
      value: 30.599637518682727
    - type: nauc_ndcg_at_3_max
      value: 36.791580459789216
    - type: nauc_ndcg_at_3_std
      value: 25.89479156863662
    - type: nauc_ndcg_at_5_diff1
      value: 31.29528680366849
    - type: nauc_ndcg_at_5_max
      value: 34.09363669130639
    - type: nauc_ndcg_at_5_std
      value: 26.748913229727943
    - type: nauc_precision_at_1000_diff1
      value: -4.271485933807063
    - type: nauc_precision_at_1000_max
      value: 25.366163422380914
    - type: nauc_precision_at_1000_std
      value: 25.530481013838568
    - type: nauc_precision_at_100_diff1
      value: 0.5495835131704634
    - type: nauc_precision_at_100_max
      value: 48.557901797757964
    - type: nauc_precision_at_100_std
      value: 41.815788332436234
    - type: nauc_precision_at_10_diff1
      value: 9.697772742135413
    - type: nauc_precision_at_10_max
      value: 47.43346995470456
    - type: nauc_precision_at_10_std
      value: 39.087209552850155
    - type: nauc_precision_at_1_diff1
      value: 37.038949494421324
    - type: nauc_precision_at_1_max
      value: 43.6843515716405
    - type: nauc_precision_at_1_std
      value: 29.062767577601583
    - type: nauc_precision_at_20_diff1
      value: 6.5884275452458185
    - type: nauc_precision_at_20_max
      value: 49.921978818717264
    - type: nauc_precision_at_20_std
      value: 41.48698751619454
    - type: nauc_precision_at_3_diff1
      value: 18.39181266067512
    - type: nauc_precision_at_3_max
      value: 47.13842403524872
    - type: nauc_precision_at_3_std
      value: 31.204774546957402
    - type: nauc_precision_at_5_diff1
      value: 14.366934091519495
    - type: nauc_precision_at_5_max
      value: 44.98856057041664
    - type: nauc_precision_at_5_std
      value: 33.86434633706037
    - type: nauc_recall_at_1000_diff1
      value: 19.132953877467653
    - type: nauc_recall_at_1000_max
      value: 26.484610396399543
    - type: nauc_recall_at_1000_std
      value: 44.59425418294402
    - type: nauc_recall_at_100_diff1
      value: 18.2269267679719
    - type: nauc_recall_at_100_max
      value: 19.23569401472271
    - type: nauc_recall_at_100_std
      value: 27.95048782794634
    - type: nauc_recall_at_10_diff1
      value: 29.636393882351232
    - type: nauc_recall_at_10_max
      value: -1.1451637872846188
    - type: nauc_recall_at_10_std
      value: 3.5050609115944673
    - type: nauc_recall_at_1_diff1
      value: 45.86732346705817
    - type: nauc_recall_at_1_max
      value: -0.7337028924656861
    - type: nauc_recall_at_1_std
      value: 3.969007656387463
    - type: nauc_recall_at_20_diff1
      value: 25.416606822860693
    - type: nauc_recall_at_20_max
      value: 3.507604434126167
    - type: nauc_recall_at_20_std
      value: 8.204428169089486
    - type: nauc_recall_at_3_diff1
      value: 33.37396491465469
    - type: nauc_recall_at_3_max
      value: 0.19079229494584185
    - type: nauc_recall_at_3_std
      value: 0.29554177247243896
    - type: nauc_recall_at_5_diff1
      value: 34.374853940101715
    - type: nauc_recall_at_5_max
      value: -0.09950975618055762
    - type: nauc_recall_at_5_std
      value: 1.3966333793032766
    - type: ndcg_at_1
      value: 32.375
    - type: ndcg_at_10
      value: 25.516
    - type: ndcg_at_100
      value: 29.213
    - type: ndcg_at_1000
      value: 36.004000000000005
    - type: ndcg_at_20
      value: 25.203999999999997
    - type: ndcg_at_3
      value: 27.889000000000003
    - type: ndcg_at_5
      value: 26.078000000000003
    - type: precision_at_1
      value: 44.5
    - type: precision_at_10
      value: 22.8
    - type: precision_at_100
      value: 7.305000000000001
    - type: precision_at_1000
      value: 1.517
    - type: precision_at_20
      value: 17.087
    - type: precision_at_3
      value: 34.083000000000006
    - type: precision_at_5
      value: 28.299999999999997
    - type: recall_at_1
      value: 4.585
    - type: recall_at_10
      value: 16.366
    - type: recall_at_100
      value: 36.771
    - type: recall_at_1000
      value: 60.239
    - type: recall_at_20
      value: 21.854000000000003
    - type: recall_at_3
      value: 8.651
    - type: recall_at_5
      value: 10.895000000000001
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB EmotionClassification (default)
      revision: 4f58c6b202a23cf9a4da393831edf4f9183cad37
      split: test
      type: mteb/emotion
    metrics:
    - type: accuracy
      value: 48.29
    - type: f1
      value: 44.290271587607116
    - type: f1_weighted
      value: 50.242229115627325
    - type: main_score
      value: 48.29
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB FEVER (default)
      revision: bea83ef9e8fb933d90a2f1d5515737465d613e12
      split: test
      type: mteb/fever
    metrics:
    - type: main_score
      value: 33.504
    - type: map_at_1
      value: 18.044
    - type: map_at_10
      value: 27.644000000000002
    - type: map_at_100
      value: 28.679
    - type: map_at_1000
      value: 28.747
    - type: map_at_20
      value: 28.244999999999997
    - type: map_at_3
      value: 24.621000000000002
    - type: map_at_5
      value: 26.262999999999998
    - type: mrr_at_1
      value: 19.33693369336934
    - type: mrr_at_10
      value: 29.386295772434227
    - type: mrr_at_100
      value: 30.393555230790692
    - type: mrr_at_1000
      value: 30.453245469699535
    - type: mrr_at_20
      value: 29.9842151301138
    - type: mrr_at_3
      value: 26.240124012401083
    - type: mrr_at_5
      value: 27.966046604660388
    - type: nauc_map_at_1000_diff1
      value: 21.197113358912173
    - type: nauc_map_at_1000_max
      value: 5.360224767609041
    - type: nauc_map_at_1000_std
      value: -7.226277841712086
    - type: nauc_map_at_100_diff1
      value: 21.193070506906462
    - type: nauc_map_at_100_max
      value: 5.356701213156949
    - type: nauc_map_at_100_std
      value: -7.234472044843086
    - type: nauc_map_at_10_diff1
      value: 21.21214854687487
    - type: nauc_map_at_10_max
      value: 5.013072982452992
    - type: nauc_map_at_10_std
      value: -7.794680387167297
    - type: nauc_map_at_1_diff1
      value: 25.128729099735956
    - type: nauc_map_at_1_max
      value: 3.207160376933345
    - type: nauc_map_at_1_std
      value: -8.943144740029735
    - type: nauc_map_at_20_diff1
      value: 21.155814570928587
    - type: nauc_map_at_20_max
      value: 5.259582088223793
    - type: nauc_map_at_20_std
      value: -7.385539946755028
    - type: nauc_map_at_3_diff1
      value: 22.103849786883735
    - type: nauc_map_at_3_max
      value: 4.351764088403068
    - type: nauc_map_at_3_std
      value: -8.59002521517384
    - type: nauc_map_at_5_diff1
      value: 21.584255518496324
    - type: nauc_map_at_5_max
      value: 4.555448236787204
    - type: nauc_map_at_5_std
      value: -8.365665226907986
    - type: nauc_mrr_at_1000_diff1
      value: 21.116177599117904
    - type: nauc_mrr_at_1000_max
      value: 5.379088881401016
    - type: nauc_mrr_at_1000_std
      value: -7.396050121471304
    - type: nauc_mrr_at_100_diff1
      value: 21.1061464811609
    - type: nauc_mrr_at_100_max
      value: 5.388746595523867
    - type: nauc_mrr_at_100_std
      value: -7.388052758239189
    - type: nauc_mrr_at_10_diff1
      value: 21.06141412827479
    - type: nauc_mrr_at_10_max
      value: 5.066514297953804
    - type: nauc_mrr_at_10_std
      value: -7.851387966112141
    - type: nauc_mrr_at_1_diff1
      value: 24.9399024874596
    - type: nauc_mrr_at_1_max
      value: 2.934019757965467
    - type: nauc_mrr_at_1_std
      value: -9.197995552521036
    - type: nauc_mrr_at_20_diff1
      value: 21.039101291564997
    - type: nauc_mrr_at_20_max
      value: 5.324147531031454
    - type: nauc_mrr_at_20_std
      value: -7.494277750700694
    - type: nauc_mrr_at_3_diff1
      value: 21.911864158855586
    - type: nauc_mrr_at_3_max
      value: 4.338076809740059
    - type: nauc_mrr_at_3_std
      value: -8.647194753014166
    - type: nauc_mrr_at_5_diff1
      value: 21.420994334374488
    - type: nauc_mrr_at_5_max
      value: 4.60819661350377
    - type: nauc_mrr_at_5_std
      value: -8.37508016803357
    - type: nauc_ndcg_at_1000_diff1
      value: 19.72912863917798
    - type: nauc_ndcg_at_1000_max
      value: 7.646491748940034
    - type: nauc_ndcg_at_1000_std
      value: -4.07147298781353
    - type: nauc_ndcg_at_100_diff1
      value: 19.611359257064237
    - type: nauc_ndcg_at_100_max
      value: 7.75610047268901
    - type: nauc_ndcg_at_100_std
      value: -4.062699446620666
    - type: nauc_ndcg_at_10_diff1
      value: 19.52738041897796
    - type: nauc_ndcg_at_10_max
      value: 6.2360420956357725
    - type: nauc_ndcg_at_10_std
      value: -6.644807690678321
    - type: nauc_ndcg_at_1_diff1
      value: 24.9399024874596
    - type: nauc_ndcg_at_1_max
      value: 2.934019757965467
    - type: nauc_ndcg_at_1_std
      value: -9.197995552521036
    - type: nauc_ndcg_at_20_diff1
      value: 19.33243817572564
    - type: nauc_ndcg_at_20_max
      value: 7.146935296531151
    - type: nauc_ndcg_at_20_std
      value: -5.175504991507281
    - type: nauc_ndcg_at_3_diff1
      value: 21.154756174209307
    - type: nauc_ndcg_at_3_max
      value: 4.713982551973281
    - type: nauc_ndcg_at_3_std
      value: -8.380199025472018
    - type: nauc_ndcg_at_5_diff1
      value: 20.324843060516955
    - type: nauc_ndcg_at_5_max
      value: 5.130345378847693
    - type: nauc_ndcg_at_5_std
      value: -7.943266710819419
    - type: nauc_precision_at_1000_diff1
      value: 0.4664007752705989
    - type: nauc_precision_at_1000_max
      value: 19.178304880632005
    - type: nauc_precision_at_1000_std
      value: 18.97537247447329
    - type: nauc_precision_at_100_diff1
      value: 8.442165363066986
    - type: nauc_precision_at_100_max
      value: 18.426727112237952
    - type: nauc_precision_at_100_std
      value: 13.668898642865269
    - type: nauc_precision_at_10_diff1
      value: 13.955990554790848
    - type: nauc_precision_at_10_max
      value: 10.114627302552769
    - type: nauc_precision_at_10_std
      value: -2.6328324881532263
    - type: nauc_precision_at_1_diff1
      value: 24.9399024874596
    - type: nauc_precision_at_1_max
      value: 2.934019757965467
    - type: nauc_precision_at_1_std
      value: -9.197995552521036
    - type: nauc_precision_at_20_diff1
      value: 11.81009397896608
    - type: nauc_precision_at_20_max
      value: 13.552268095662074
    - type: nauc_precision_at_20_std
      value: 3.40206785511483
    - type: nauc_precision_at_3_diff1
      value: 18.571494545732914
    - type: nauc_precision_at_3_max
      value: 5.863463077194485
    - type: nauc_precision_at_3_std
      value: -7.616080429294618
    - type: nauc_precision_at_5_diff1
      value: 16.529672676412613
    - type: nauc_precision_at_5_max
      value: 6.739757756034394
    - type: nauc_precision_at_5_std
      value: -6.696702432485899
    - type: nauc_recall_at_1000_diff1
      value: 8.21901292719372
    - type: nauc_recall_at_1000_max
      value: 24.727650913551372
    - type: nauc_recall_at_1000_std
      value: 27.25469920285314
    - type: nauc_recall_at_100_diff1
      value: 12.1659312543603
    - type: nauc_recall_at_100_max
      value: 18.333798430261503
    - type: nauc_recall_at_100_std
      value: 13.073233154192657
    - type: nauc_recall_at_10_diff1
      value: 14.210250438173627
    - type: nauc_recall_at_10_max
      value: 9.272403472015942
    - type: nauc_recall_at_10_std
      value: -3.0236654339158395
    - type: nauc_recall_at_1_diff1
      value: 25.128729099735956
    - type: nauc_recall_at_1_max
      value: 3.207160376933345
    - type: nauc_recall_at_1_std
      value: -8.943144740029735
    - type: nauc_recall_at_20_diff1
      value: 12.949168944136929
    - type: nauc_recall_at_20_max
      value: 12.842070355653451
    - type: nauc_recall_at_20_std
      value: 2.869920583633798
    - type: nauc_recall_at_3_diff1
      value: 18.496377801915955
    - type: nauc_recall_at_3_max
      value: 5.467087584819722
    - type: nauc_recall_at_3_std
      value: -7.56424448936203
    - type: nauc_recall_at_5_diff1
      value: 16.66309779367226
    - type: nauc_recall_at_5_max
      value: 6.313983077243061
    - type: nauc_recall_at_5_std
      value: -6.606213139129608
    - type: ndcg_at_1
      value: 19.337
    - type: ndcg_at_10
      value: 33.504
    - type: ndcg_at_100
      value: 38.68
    - type: ndcg_at_1000
      value: 40.474
    - type: ndcg_at_20
      value: 35.663
    - type: ndcg_at_3
      value: 27.232
    - type: ndcg_at_5
      value: 30.177
    - type: precision_at_1
      value: 19.337
    - type: precision_at_10
      value: 5.473
    - type: precision_at_100
      value: 0.8250000000000001
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_20
      value: 3.212
    - type: precision_at_3
      value: 11.901
    - type: precision_at_5
      value: 8.713
    - type: recall_at_1
      value: 18.044
    - type: recall_at_10
      value: 50.26199999999999
    - type: recall_at_100
      value: 74.25
    - type: recall_at_1000
      value: 87.905
    - type: recall_at_20
      value: 58.550999999999995
    - type: recall_at_3
      value: 33.161
    - type: recall_at_5
      value: 40.198
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB FiQA2018 (default)
      revision: 27a168819829fe9bcd655c2df245fb19452e8e06
      split: test
      type: mteb/fiqa
    metrics:
    - type: main_score
      value: 17.976
    - type: map_at_1
      value: 7.792000000000001
    - type: map_at_10
      value: 13.0
    - type: map_at_100
      value: 14.100999999999999
    - type: map_at_1000
      value: 14.301
    - type: map_at_20
      value: 13.52
    - type: map_at_3
      value: 10.843
    - type: map_at_5
      value: 11.878
    - type: mrr_at_1
      value: 15.895061728395063
    - type: mrr_at_10
      value: 22.31193660591808
    - type: mrr_at_100
      value: 23.2877699627266
    - type: mrr_at_1000
      value: 23.381067507259456
    - type: mrr_at_20
      value: 22.827454136542826
    - type: mrr_at_3
      value: 19.88168724279835
    - type: mrr_at_5
      value: 20.961934156378597
    - type: nauc_map_at_1000_diff1
      value: 26.692130203802293
    - type: nauc_map_at_1000_max
      value: 6.4075849334836565
    - type: nauc_map_at_1000_std
      value: -3.73275024757571
    - type: nauc_map_at_100_diff1
      value: 26.632161984458726
    - type: nauc_map_at_100_max
      value: 6.2287093296309335
    - type: nauc_map_at_100_std
      value: -3.846917465734305
    - type: nauc_map_at_10_diff1
      value: 26.82793337901525
    - type: nauc_map_at_10_max
      value: 5.964060354762653
    - type: nauc_map_at_10_std
      value: -4.866678549363916
    - type: nauc_map_at_1_diff1
      value: 34.384366407953124
    - type: nauc_map_at_1_max
      value: 9.604169568417987
    - type: nauc_map_at_1_std
      value: -6.260955282260347
    - type: nauc_map_at_20_diff1
      value: 26.93171611922343
    - type: nauc_map_at_20_max
      value: 5.834195409943979
    - type: nauc_map_at_20_std
      value: -4.681006600256935
    - type: nauc_map_at_3_diff1
      value: 28.36058264123047
    - type: nauc_map_at_3_max
      value: 7.508527545122337
    - type: nauc_map_at_3_std
      value: -4.988672072096864
    - type: nauc_map_at_5_diff1
      value: 27.2345922474392
    - type: nauc_map_at_5_max
      value: 6.233631918196583
    - type: nauc_map_at_5_std
      value: -5.591302232606139
    - type: nauc_mrr_at_1000_diff1
      value: 25.0035885940064
    - type: nauc_mrr_at_1000_max
      value: 6.896052718656624
    - type: nauc_mrr_at_1000_std
      value: -6.481600423055105
    - type: nauc_mrr_at_100_diff1
      value: 24.993122116083757
    - type: nauc_mrr_at_100_max
      value: 6.837774558302377
    - type: nauc_mrr_at_100_std
      value: -6.476270787729837
    - type: nauc_mrr_at_10_diff1
      value: 24.986704881829326
    - type: nauc_mrr_at_10_max
      value: 6.730638735416298
    - type: nauc_mrr_at_10_std
      value: -6.875384862951013
    - type: nauc_mrr_at_1_diff1
      value: 30.757462704144473
    - type: nauc_mrr_at_1_max
      value: 9.494036047978879
    - type: nauc_mrr_at_1_std
      value: -8.55327939175485
    - type: nauc_mrr_at_20_diff1
      value: 25.066582206134203
    - type: nauc_mrr_at_20_max
      value: 6.640370084188472
    - type: nauc_mrr_at_20_std
      value: -6.861230381817542
    - type: nauc_mrr_at_3_diff1
      value: 24.700095299205675
    - type: nauc_mrr_at_3_max
      value: 6.81056900129325
    - type: nauc_mrr_at_3_std
      value: -7.54289500858466
    - type: nauc_mrr_at_5_diff1
      value: 24.911830324957428
    - type: nauc_mrr_at_5_max
      value: 6.487419609168333
    - type: nauc_mrr_at_5_std
      value: -7.559191642416501
    - type: nauc_ndcg_at_1000_diff1
      value: 24.271059266102714
    - type: nauc_ndcg_at_1000_max
      value: 9.036727250049996
    - type: nauc_ndcg_at_1000_std
      value: 0.9146422915784614
    - type: nauc_ndcg_at_100_diff1
      value: 23.3906476337681
    - type: nauc_ndcg_at_100_max
      value: 6.666510188169236
    - type: nauc_ndcg_at_100_std
      value: -0.13425031252447506
    - type: nauc_ndcg_at_10_diff1
      value: 24.467859958572237
    - type: nauc_ndcg_at_10_max
      value: 5.180189255703774
    - type: nauc_ndcg_at_10_std
      value: -4.564295424146644
    - type: nauc_ndcg_at_1_diff1
      value: 30.757462704144473
    - type: nauc_ndcg_at_1_max
      value: 9.494036047978879
    - type: nauc_ndcg_at_1_std
      value: -8.55327939175485
    - type: nauc_ndcg_at_20_diff1
      value: 24.423852550652708
    - type: nauc_ndcg_at_20_max
      value: 4.621408762131097
    - type: nauc_ndcg_at_20_std
      value: -4.174045549905428
    - type: nauc_ndcg_at_3_diff1
      value: 25.581953443217888
    - type: nauc_ndcg_at_3_max
      value: 7.124469256934023
    - type: nauc_ndcg_at_3_std
      value: -6.179824512286984
    - type: nauc_ndcg_at_5_diff1
      value: 24.99310386834495
    - type: nauc_ndcg_at_5_max
      value: 5.556772674359943
    - type: nauc_ndcg_at_5_std
      value: -6.44181066458889
    - type: nauc_precision_at_1000_diff1
      value: 9.05364142609685
    - type: nauc_precision_at_1000_max
      value: 15.91160424741351
    - type: nauc_precision_at_1000_std
      value: 5.9579504982280795
    - type: nauc_precision_at_100_diff1
      value: 14.495341408480996
    - type: nauc_precision_at_100_max
      value: 10.619524667734845
    - type: nauc_precision_at_100_std
      value: 6.94473626177151
    - type: nauc_precision_at_10_diff1
      value: 17.681863001899455
    - type: nauc_precision_at_10_max
      value: 2.933379217123649
    - type: nauc_precision_at_10_std
      value: -6.189549061252104
    - type: nauc_precision_at_1_diff1
      value: 30.757462704144473
    - type: nauc_precision_at_1_max
      value: 9.494036047978879
    - type: nauc_precision_at_1_std
      value: -8.55327939175485
    - type: nauc_precision_at_20_diff1
      value: 16.85819808382462
    - type: nauc_precision_at_20_max
      value: 1.8703103333361615
    - type: nauc_precision_at_20_std
      value: -4.095334243709078
    - type: nauc_precision_at_3_diff1
      value: 21.936934551805678
    - type: nauc_precision_at_3_max
      value: 5.700969823325045
    - type: nauc_precision_at_3_std
      value: -7.50541930072883
    - type: nauc_precision_at_5_diff1
      value: 20.4629322334308
    - type: nauc_precision_at_5_max
      value: 3.570606410878902
    - type: nauc_precision_at_5_std
      value: -8.072847794719719
    - type: nauc_recall_at_1000_diff1
      value: 12.640253594366952
    - type: nauc_recall_at_1000_max
      value: 15.416517388620205
    - type: nauc_recall_at_1000_std
      value: 22.104957683222754
    - type: nauc_recall_at_100_diff1
      value: 11.611361782989556
    - type: nauc_recall_at_100_max
      value: 5.315964969533011
    - type: nauc_recall_at_100_std
      value: 10.76920407330154
    - type: nauc_recall_at_10_diff1
      value: 17.641989640006983
    - type: nauc_recall_at_10_max
      value: 2.506101915806224
    - type: nauc_recall_at_10_std
      value: -1.405488559040536
    - type: nauc_recall_at_1_diff1
      value: 34.384366407953124
    - type: nauc_recall_at_1_max
      value: 9.604169568417987
    - type: nauc_recall_at_1_std
      value: -6.260955282260347
    - type: nauc_recall_at_20_diff1
      value: 17.306971008788068
    - type: nauc_recall_at_20_max
      value: 1.178463149677817
    - type: nauc_recall_at_20_std
      value: -1.2622013006761976
    - type: nauc_recall_at_3_diff1
      value: 21.599954957857086
    - type: nauc_recall_at_3_max
      value: 4.5988093302070405
    - type: nauc_recall_at_3_std
      value: -4.111196297634325
    - type: nauc_recall_at_5_diff1
      value: 18.40790560459866
    - type: nauc_recall_at_5_max
      value: 2.227515887769542
    - type: nauc_recall_at_5_std
      value: -5.122348436923138
    - type: ndcg_at_1
      value: 15.895000000000001
    - type: ndcg_at_10
      value: 17.976
    - type: ndcg_at_100
      value: 23.543
    - type: ndcg_at_1000
      value: 27.942
    - type: ndcg_at_20
      value: 19.73
    - type: ndcg_at_3
      value: 14.652000000000001
    - type: ndcg_at_5
      value: 15.581
    - type: precision_at_1
      value: 15.895000000000001
    - type: precision_at_10
      value: 5.2780000000000005
    - type: precision_at_100
      value: 1.0710000000000002
    - type: precision_at_1000
      value: 0.185
    - type: precision_at_20
      value: 3.3099999999999996
    - type: precision_at_3
      value: 9.774
    - type: precision_at_5
      value: 7.438000000000001
    - type: recall_at_1
      value: 7.792000000000001
    - type: recall_at_10
      value: 23.491999999999997
    - type: recall_at_100
      value: 46.01
    - type: recall_at_1000
      value: 72.858
    - type: recall_at_20
      value: 29.262
    - type: recall_at_3
      value: 13.442000000000002
    - type: recall_at_5
      value: 16.885
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB HotpotQA (default)
      revision: ab518f4d6fcca38d87c25209f94beba119d02014
      split: test
      type: mteb/hotpotqa
    metrics:
    - type: main_score
      value: 40.216
    - type: map_at_1
      value: 23.16
    - type: map_at_10
      value: 32.046
    - type: map_at_100
      value: 33.0
    - type: map_at_1000
      value: 33.102
    - type: map_at_20
      value: 32.568000000000005
    - type: map_at_3
      value: 29.654999999999998
    - type: map_at_5
      value: 31.011
    - type: mrr_at_1
      value: 46.320054017555705
    - type: mrr_at_10
      value: 53.95531762108416
    - type: mrr_at_100
      value: 54.57200329952687
    - type: mrr_at_1000
      value: 54.613276611506464
    - type: mrr_at_20
      value: 54.31181386260544
    - type: mrr_at_3
      value: 52.09318028359241
    - type: mrr_at_5
      value: 53.16205266711693
    - type: nauc_map_at_1000_diff1
      value: 44.859067312276274
    - type: nauc_map_at_1000_max
      value: 25.916972606593696
    - type: nauc_map_at_1000_std
      value: 6.253445218067477
    - type: nauc_map_at_100_diff1
      value: 44.8632434061937
    - type: nauc_map_at_100_max
      value: 25.909099999441942
    - type: nauc_map_at_100_std
      value: 6.214788111779019
    - type: nauc_map_at_10_diff1
      value: 45.30759397690741
    - type: nauc_map_at_10_max
      value: 25.690927362066233
    - type: nauc_map_at_10_std
      value: 5.285259083844247
    - type: nauc_map_at_1_diff1
      value: 61.196409961814155
    - type: nauc_map_at_1_max
      value: 28.72930920023417
    - type: nauc_map_at_1_std
      value: 0.5023892901280935
    - type: nauc_map_at_20_diff1
      value: 45.03381245393179
    - type: nauc_map_at_20_max
      value: 25.80883711775135
    - type: nauc_map_at_20_std
      value: 5.813602144954199
    - type: nauc_map_at_3_diff1
      value: 47.606878718192625
    - type: nauc_map_at_3_max
      value: 26.25063014273412
    - type: nauc_map_at_3_std
      value: 3.8842714485913095
    - type: nauc_map_at_5_diff1
      value: 45.969349613068815
    - type: nauc_map_at_5_max
      value: 25.90509255737118
    - type: nauc_map_at_5_std
      value: 4.445890407273527
    - type: nauc_mrr_at_1000_diff1
      value: 58.09795156596312
    - type: nauc_mrr_at_1000_max
      value: 28.293914309719643
    - type: nauc_mrr_at_1000_std
      value: 2.9543852693821604
    - type: nauc_mrr_at_100_diff1
      value: 58.08973285225233
    - type: nauc_mrr_at_100_max
      value: 28.296027316962384
    - type: nauc_mrr_at_100_std
      value: 2.9669718825797387
    - type: nauc_mrr_at_10_diff1
      value: 58.1296976627478
    - type: nauc_mrr_at_10_max
      value: 28.214896438043247
    - type: nauc_mrr_at_10_std
      value: 2.6724211845588295
    - type: nauc_mrr_at_1_diff1
      value: 61.196409961814155
    - type: nauc_mrr_at_1_max
      value: 28.72930920023417
    - type: nauc_mrr_at_1_std
      value: 0.5023892901280935
    - type: nauc_mrr_at_20_diff1
      value: 58.07901672129855
    - type: nauc_mrr_at_20_max
      value: 28.261220041773193
    - type: nauc_mrr_at_20_std
      value: 2.8754516882066112
    - type: nauc_mrr_at_3_diff1
      value: 58.49251126681858
    - type: nauc_mrr_at_3_max
      value: 28.330048816492358
    - type: nauc_mrr_at_3_std
      value: 2.088307118099806
    - type: nauc_mrr_at_5_diff1
      value: 58.293498499452056
    - type: nauc_mrr_at_5_max
      value: 28.254147613755787
    - type: nauc_mrr_at_5_std
      value: 2.317453426169504
    - type: nauc_ndcg_at_1000_diff1
      value: 44.74813748409359
    - type: nauc_ndcg_at_1000_max
      value: 26.540352967644417
    - type: nauc_ndcg_at_1000_std
      value: 10.102814146840283
    - type: nauc_ndcg_at_100_diff1
      value: 44.87337180897993
    - type: nauc_ndcg_at_100_max
      value: 26.474465177673085
    - type: nauc_ndcg_at_100_std
      value: 9.579371689924795
    - type: nauc_ndcg_at_10_diff1
      value: 46.581223548767
    - type: nauc_ndcg_at_10_max
      value: 25.828889039362295
    - type: nauc_ndcg_at_10_std
      value: 6.0964295503096295
    - type: nauc_ndcg_at_1_diff1
      value: 61.196409961814155
    - type: nauc_ndcg_at_1_max
      value: 28.72930920023417
    - type: nauc_ndcg_at_1_std
      value: 0.5023892901280935
    - type: nauc_ndcg_at_20_diff1
      value: 45.77866051784014
    - type: nauc_ndcg_at_20_max
      value: 26.088161426584833
    - type: nauc_ndcg_at_20_std
      value: 7.520411389641454
    - type: nauc_ndcg_at_3_diff1
      value: 49.79589333047773
    - type: nauc_ndcg_at_3_max
      value: 26.5918118122357
    - type: nauc_ndcg_at_3_std
      value: 3.8100097440907477
    - type: nauc_ndcg_at_5_diff1
      value: 47.7687125131952
    - type: nauc_ndcg_at_5_max
      value: 26.124001776821682
    - type: nauc_ndcg_at_5_std
      value: 4.552169333444345
    - type: nauc_precision_at_1000_diff1
      value: 9.937767084200962
    - type: nauc_precision_at_1000_max
      value: 17.821085008921447
    - type: nauc_precision_at_1000_std
      value: 30.491638472583794
    - type: nauc_precision_at_100_diff1
      value: 20.162873612867944
    - type: nauc_precision_at_100_max
      value: 20.353909763458464
    - type: nauc_precision_at_100_std
      value: 23.235976792350037
    - type: nauc_precision_at_10_diff1
      value: 32.88560093643963
    - type: nauc_precision_at_10_max
      value: 21.411457505696205
    - type: nauc_precision_at_10_std
      value: 10.425552179784567
    - type: nauc_precision_at_1_diff1
      value: 61.196409961814155
    - type: nauc_precision_at_1_max
      value: 28.72930920023417
    - type: nauc_precision_at_1_std
      value: 0.5023892901280935
    - type: nauc_precision_at_20_diff1
      value: 28.258115065654987
    - type: nauc_precision_at_20_max
      value: 20.961811561439706
    - type: nauc_precision_at_20_std
      value: 14.46899018450582
    - type: nauc_precision_at_3_diff1
      value: 42.7620046846948
    - type: nauc_precision_at_3_max
      value: 24.705286955229948
    - type: nauc_precision_at_3_std
      value: 5.457836377466167
    - type: nauc_precision_at_5_diff1
      value: 37.60627429610388
    - type: nauc_precision_at_5_max
      value: 23.124944762838144
    - type: nauc_precision_at_5_std
      value: 6.786960218266853
    - type: nauc_recall_at_1000_diff1
      value: 9.937767084201072
    - type: nauc_recall_at_1000_max
      value: 17.821085008921614
    - type: nauc_recall_at_1000_std
      value: 30.491638472583965
    - type: nauc_recall_at_100_diff1
      value: 20.16287361286792
    - type: nauc_recall_at_100_max
      value: 20.353909763458358
    - type: nauc_recall_at_100_std
      value: 23.23597679234997
    - type: nauc_recall_at_10_diff1
      value: 32.88560093643965
    - type: nauc_recall_at_10_max
      value: 21.41145750569624
    - type: nauc_recall_at_10_std
      value: 10.425552179784642
    - type: nauc_recall_at_1_diff1
      value: 61.196409961814155
    - type: nauc_recall_at_1_max
      value: 28.72930920023417
    - type: nauc_recall_at_1_std
      value: 0.5023892901280935
    - type: nauc_recall_at_20_diff1
      value: 28.25811506565502
    - type: nauc_recall_at_20_max
      value: 20.961811561439756
    - type: nauc_recall_at_20_std
      value: 14.468990184505875
    - type: nauc_recall_at_3_diff1
      value: 42.76200468469484
    - type: nauc_recall_at_3_max
      value: 24.705286955229923
    - type: nauc_recall_at_3_std
      value: 5.4578363774661485
    - type: nauc_recall_at_5_diff1
      value: 37.60627429610391
    - type: nauc_recall_at_5_max
      value: 23.124944762838098
    - type: nauc_recall_at_5_std
      value: 6.786960218266914
    - type: ndcg_at_1
      value: 46.32
    - type: ndcg_at_10
      value: 40.216
    - type: ndcg_at_100
      value: 44.330000000000005
    - type: ndcg_at_1000
      value: 46.656
    - type: ndcg_at_20
      value: 41.787
    - type: ndcg_at_3
      value: 35.998000000000005
    - type: ndcg_at_5
      value: 38.089
    - type: precision_at_1
      value: 46.32
    - type: precision_at_10
      value: 8.641
    - type: precision_at_100
      value: 1.191
    - type: precision_at_1000
      value: 0.15
    - type: precision_at_20
      value: 4.828
    - type: precision_at_3
      value: 22.467000000000002
    - type: precision_at_5
      value: 15.137999999999998
    - type: recall_at_1
      value: 23.16
    - type: recall_at_10
      value: 43.207
    - type: recall_at_100
      value: 59.553999999999995
    - type: recall_at_1000
      value: 75.03699999999999
    - type: recall_at_20
      value: 48.285
    - type: recall_at_3
      value: 33.7
    - type: recall_at_5
      value: 37.846000000000004
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB ImdbClassification (default)
      revision: 3d86128a09e091d6018b6d26cad27f2739fc2db7
      split: test
      type: mteb/imdb
    metrics:
    - type: accuracy
      value: 70.126
    - type: ap
      value: 64.38953964700043
    - type: ap_weighted
      value: 64.38953964700043
    - type: f1
      value: 69.92812220701516
    - type: f1_weighted
      value: 69.92812220701516
    - type: main_score
      value: 70.126
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB MSMARCO (default)
      revision: c5a29a104738b98a9e76336939199e264163d4a0
      split: test
      type: mteb/msmarco
    metrics:
    - type: main_score
      value: 37.678
    - type: map_at_1
      value: 1.3
    - type: map_at_10
      value: 6.458
    - type: map_at_100
      value: 18.192
    - type: map_at_1000
      value: 22.869
    - type: map_at_20
      value: 9.508999999999999
    - type: map_at_3
      value: 2.7449999999999997
    - type: map_at_5
      value: 3.9989999999999997
    - type: mrr_at_1
      value: 67.44186046511628
    - type: mrr_at_10
      value: 76.70542635658914
    - type: mrr_at_100
      value: 76.75490681180933
    - type: mrr_at_1000
      value: 76.75897251354947
    - type: mrr_at_20
      value: 76.70542635658914
    - type: mrr_at_3
      value: 74.03100775193799
    - type: mrr_at_5
      value: 76.70542635658914
    - type: nauc_map_at_1000_diff1
      value: -9.887545588069491
    - type: nauc_map_at_1000_max
      value: 64.03933042840461
    - type: nauc_map_at_1000_std
      value: 65.23188262074746
    - type: nauc_map_at_100_diff1
      value: -10.206004224159292
    - type: nauc_map_at_100_max
      value: 56.69013155432794
    - type: nauc_map_at_100_std
      value: 59.28810375228662
    - type: nauc_map_at_10_diff1
      value: -17.04940102245024
    - type: nauc_map_at_10_max
      value: 27.273929641443555
    - type: nauc_map_at_10_std
      value: 27.82349143703665
    - type: nauc_map_at_1_diff1
      value: -26.202958880643145
    - type: nauc_map_at_1_max
      value: -7.124386540229982
    - type: nauc_map_at_1_std
      value: 0.6951328400074942
    - type: nauc_map_at_20_diff1
      value: -14.164361766078779
    - type: nauc_map_at_20_max
      value: 35.702893018714846
    - type: nauc_map_at_20_std
      value: 39.30480590474426
    - type: nauc_map_at_3_diff1
      value: -25.744666153328176
    - type: nauc_map_at_3_max
      value: 9.001802346350422
    - type: nauc_map_at_3_std
      value: 14.553440561965308
    - type: nauc_map_at_5_diff1
      value: -22.50484947409843
    - type: nauc_map_at_5_max
      value: 18.57285643186047
    - type: nauc_map_at_5_std
      value: 22.228488573704613
    - type: nauc_mrr_at_1000_diff1
      value: -47.733315106072844
    - type: nauc_mrr_at_1000_max
      value: 44.821461988966924
    - type: nauc_mrr_at_1000_std
      value: 47.84242487878596
    - type: nauc_mrr_at_100_diff1
      value: -47.727366114165626
    - type: nauc_mrr_at_100_max
      value: 44.83250848852219
    - type: nauc_mrr_at_100_std
      value: 47.852866594915966
    - type: nauc_mrr_at_10_diff1
      value: -47.88635303336966
    - type: nauc_mrr_at_10_max
      value: 44.96659313758439
    - type: nauc_mrr_at_10_std
      value: 47.8236450194703
    - type: nauc_mrr_at_1_diff1
      value: -39.17983238638767
    - type: nauc_mrr_at_1_max
      value: 39.6591497254911
    - type: nauc_mrr_at_1_std
      value: 37.90542432955914
    - type: nauc_mrr_at_20_diff1
      value: -47.88635303336966
    - type: nauc_mrr_at_20_max
      value: 44.96659313758439
    - type: nauc_mrr_at_20_std
      value: 47.8236450194703
    - type: nauc_mrr_at_3_diff1
      value: -46.2121294259396
    - type: nauc_mrr_at_3_max
      value: 44.10697671384829
    - type: nauc_mrr_at_3_std
      value: 49.383440451932195
    - type: nauc_mrr_at_5_diff1
      value: -47.88635303336966
    - type: nauc_mrr_at_5_max
      value: 44.96659313758439
    - type: nauc_mrr_at_5_std
      value: 47.8236450194703
    - type: nauc_ndcg_at_1000_diff1
      value: -28.786349342559653
    - type: nauc_ndcg_at_1000_max
      value: 59.97264099708227
    - type: nauc_ndcg_at_1000_std
      value: 68.20776309895734
    - type: nauc_ndcg_at_100_diff1
      value: -15.688954074018316
    - type: nauc_ndcg_at_100_max
      value: 60.23683014554728
    - type: nauc_ndcg_at_100_std
      value: 63.11339418621779
    - type: nauc_ndcg_at_10_diff1
      value: -24.90231794908744
    - type: nauc_ndcg_at_10_max
      value: 53.375692071166156
    - type: nauc_ndcg_at_10_std
      value: 52.48592330227771
    - type: nauc_ndcg_at_1_diff1
      value: -21.653521280252825
    - type: nauc_ndcg_at_1_max
      value: 14.274727299758839
    - type: nauc_ndcg_at_1_std
      value: 20.292187616514617
    - type: nauc_ndcg_at_20_diff1
      value: -22.118549404986005
    - type: nauc_ndcg_at_20_max
      value: 59.42642999251343
    - type: nauc_ndcg_at_20_std
      value: 60.036912592320654
    - type: nauc_ndcg_at_3_diff1
      value: -28.0998335530241
    - type: nauc_ndcg_at_3_max
      value: 37.19495998818337
    - type: nauc_ndcg_at_3_std
      value: 45.602451446392834
    - type: nauc_ndcg_at_5_diff1
      value: -29.3466175706748
    - type: nauc_ndcg_at_5_max
      value: 47.619470848266225
    - type: nauc_ndcg_at_5_std
      value: 53.344706345912115
    - type: nauc_precision_at_1000_diff1
      value: -6.998782483418799
    - type: nauc_precision_at_1000_max
      value: 53.11312314325285
    - type: nauc_precision_at_1000_std
      value: 48.25955635434035
    - type: nauc_precision_at_100_diff1
      value: -4.605735408234786
    - type: nauc_precision_at_100_max
      value: 65.51127106580617
    - type: nauc_precision_at_100_std
      value: 60.73394086559162
    - type: nauc_precision_at_10_diff1
      value: -20.30338592030741
    - type: nauc_precision_at_10_max
      value: 68.4564447240256
    - type: nauc_precision_at_10_std
      value: 58.60683771481885
    - type: nauc_precision_at_1_diff1
      value: -39.17983238638767
    - type: nauc_precision_at_1_max
      value: 39.6591497254911
    - type: nauc_precision_at_1_std
      value: 37.90542432955914
    - type: nauc_precision_at_20_diff1
      value: -17.26105647691607
    - type: nauc_precision_at_20_max
      value: 66.01915424661092
    - type: nauc_precision_at_20_std
      value: 60.93952502719073
    - type: nauc_precision_at_3_diff1
      value: -31.21293042037448
    - type: nauc_precision_at_3_max
      value: 62.24093293515494
    - type: nauc_precision_at_3_std
      value: 60.55027014842964
    - type: nauc_precision_at_5_diff1
      value: -29.57322674528564
    - type: nauc_precision_at_5_max
      value: 67.67852068990298
    - type: nauc_precision_at_5_std
      value: 62.04281793717718
    - type: nauc_recall_at_1000_diff1
      value: -32.248423680290855
    - type: nauc_recall_at_1000_max
      value: 50.18123492869779
    - type: nauc_recall_at_1000_std
      value: 60.66089355852352
    - type: nauc_recall_at_100_diff1
      value: -14.301809847601831
    - type: nauc_recall_at_100_max
      value: 44.515754859360854
    - type: nauc_recall_at_100_std
      value: 49.81279937525628
    - type: nauc_recall_at_10_diff1
      value: -16.93047529431789
    - type: nauc_recall_at_10_max
      value: 20.210519584735927
    - type: nauc_recall_at_10_std
      value: 21.352140573597637
    - type: nauc_recall_at_1_diff1
      value: -26.202958880643145
    - type: nauc_recall_at_1_max
      value: -7.124386540229982
    - type: nauc_recall_at_1_std
      value: 0.6951328400074942
    - type: nauc_recall_at_20_diff1
      value: -17.54223589626273
    - type: nauc_recall_at_20_max
      value: 27.07050603517775
    - type: nauc_recall_at_20_std
      value: 32.05685363558265
    - type: nauc_recall_at_3_diff1
      value: -23.618200725056937
    - type: nauc_recall_at_3_max
      value: 4.640818564571053
    - type: nauc_recall_at_3_std
      value: 8.168671193501167
    - type: nauc_recall_at_5_diff1
      value: -21.796040536265913
    - type: nauc_recall_at_5_max
      value: 12.040922147593534
    - type: nauc_recall_at_5_std
      value: 15.749947387697357
    - type: ndcg_at_1
      value: 44.186
    - type: ndcg_at_10
      value: 37.678
    - type: ndcg_at_100
      value: 37.508
    - type: ndcg_at_1000
      value: 46.955999999999996
    - type: ndcg_at_20
      value: 35.888999999999996
    - type: ndcg_at_3
      value: 39.298
    - type: ndcg_at_5
      value: 39.582
    - type: precision_at_1
      value: 67.44200000000001
    - type: precision_at_10
      value: 47.674
    - type: precision_at_100
      value: 25.0
    - type: precision_at_1000
      value: 5.202
    - type: precision_at_20
      value: 41.047
    - type: precision_at_3
      value: 55.814
    - type: precision_at_5
      value: 53.952999999999996
    - type: recall_at_1
      value: 1.3
    - type: recall_at_10
      value: 8.068999999999999
    - type: recall_at_100
      value: 32.096000000000004
    - type: recall_at_1000
      value: 59.51499999999999
    - type: recall_at_20
      value: 12.834000000000001
    - type: recall_at_3
      value: 3.056
    - type: recall_at_5
      value: 4.806
    task:
      type: Retrieval
  - dataset:
      config: en
      name: MTEB MTOPDomainClassification (en)
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
      split: test
      type: mteb/mtop_domain
    metrics:
    - type: accuracy
      value: 90.6110351117191
    - type: f1
      value: 89.7664246079241
    - type: f1_weighted
      value: 90.6866222833887
    - type: main_score
      value: 90.6110351117191
    task:
      type: Classification
  - dataset:
      config: en
      name: MTEB MTOPIntentClassification (en)
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
      split: test
      type: mteb/mtop_intent
    metrics:
    - type: accuracy
      value: 59.90424076607387
    - type: f1
      value: 40.877183239388046
    - type: f1_weighted
      value: 63.17863037028273
    - type: main_score
      value: 59.90424076607387
    task:
      type: Classification
  - dataset:
      config: en
      name: MTEB MassiveIntentClassification (en)
      revision: 4672e20407010da34463acc759c162ca9734bca6
      split: test
      type: mteb/amazon_massive_intent
    metrics:
    - type: accuracy
      value: 65.137861466039
    - type: f1
      value: 62.90244115213582
    - type: f1_weighted
      value: 64.11200345839086
    - type: main_score
      value: 65.137861466039
    task:
      type: Classification
  - dataset:
      config: en
      name: MTEB MassiveScenarioClassification (en)
      revision: fad2c6e8459f9e1c45d9315f4953d921437d70f8
      split: test
      type: mteb/amazon_massive_scenario
    metrics:
    - type: accuracy
      value: 71.38870208473436
    - type: f1
      value: 70.05482587512654
    - type: f1_weighted
      value: 71.25688112025705
    - type: main_score
      value: 71.38870208473436
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB MedrxivClusteringP2P (default)
      revision: e7a26af6f3ae46b30dde8737f02c07b1505bcc73
      split: test
      type: mteb/medrxiv-clustering-p2p
    metrics:
    - type: main_score
      value: 29.317315039846775
    - type: v_measure
      value: 29.317315039846775
    - type: v_measure_std
      value: 1.5543590443494053
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB MedrxivClusteringS2S (default)
      revision: 35191c8c0dca72d8ff3efcd72aa802307d469663
      split: test
      type: mteb/medrxiv-clustering-s2s
    metrics:
    - type: main_score
      value: 25.68391422034485
    - type: v_measure
      value: 25.68391422034485
    - type: v_measure_std
      value: 1.729769358006023
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB MindSmallReranking (default)
      revision: 59042f120c80e8afa9cdbb224f67076cec0fc9a7
      split: test
      type: mteb/mind_small
    metrics:
    - type: main_score
      value: 30.666907553958655
    - type: map
      value: 30.666907553958655
    - type: mrr
      value: 31.595166716117408
    - type: nAUC_map_diff1
      value: 9.04433618641072
    - type: nAUC_map_max
      value: -22.25112566123981
    - type: nAUC_map_std
      value: -7.843708225558461
    - type: nAUC_mrr_diff1
      value: 9.290814967375317
    - type: nAUC_mrr_max
      value: -16.507108590097268
    - type: nAUC_mrr_std
      value: -4.804937846268291
    task:
      type: Reranking
  - dataset:
      config: default
      name: MTEB NFCorpus (default)
      revision: ec0fa4fe99da2ff19ca1214b7966684033a58814
      split: test
      type: mteb/nfcorpus
    metrics:
    - type: main_score
      value: 27.16
    - type: map_at_1
      value: 4.561
    - type: map_at_10
      value: 9.27
    - type: map_at_100
      value: 11.644
    - type: map_at_1000
      value: 12.842999999999998
    - type: map_at_20
      value: 10.327
    - type: map_at_3
      value: 7.053
    - type: map_at_5
      value: 8.037999999999998
    - type: mrr_at_1
      value: 38.080495356037154
    - type: mrr_at_10
      value: 47.06840630989237
    - type: mrr_at_100
      value: 47.74539552696533
    - type: mrr_at_1000
      value: 47.79398428238601
    - type: mrr_at_20
      value: 47.39011213236981
    - type: mrr_at_3
      value: 44.94324045407637
    - type: mrr_at_5
      value: 46.08875128998968
    - type: nauc_map_at_1000_diff1
      value: 40.4041423598124
    - type: nauc_map_at_1000_max
      value: 28.030317236467305
    - type: nauc_map_at_1000_std
      value: 10.127441487340901
    - type: nauc_map_at_100_diff1
      value: 42.11665366054116
    - type: nauc_map_at_100_max
      value: 26.21149470991564
    - type: nauc_map_at_100_std
      value: 6.062707293500967
    - type: nauc_map_at_10_diff1
      value: 46.93633791543067
    - type: nauc_map_at_10_max
      value: 21.297836855982794
    - type: nauc_map_at_10_std
      value: -1.1914488274665422
    - type: nauc_map_at_1_diff1
      value: 50.61355369486035
    - type: nauc_map_at_1_max
      value: 7.072899231939653
    - type: nauc_map_at_1_std
      value: -8.829084633669
    - type: nauc_map_at_20_diff1
      value: 44.91818163266758
    - type: nauc_map_at_20_max
      value: 24.07822822706987
    - type: nauc_map_at_20_std
      value: 2.199092642576738
    - type: nauc_map_at_3_diff1
      value: 48.80951904378301
    - type: nauc_map_at_3_max
      value: 15.221884690575532
    - type: nauc_map_at_3_std
      value: -6.438393157982664
    - type: nauc_map_at_5_diff1
      value: 48.314911451618755
    - type: nauc_map_at_5_max
      value: 17.54244302114833
    - type: nauc_map_at_5_std
      value: -4.6416655677428915
    - type: nauc_mrr_at_1000_diff1
      value: 39.63070814105974
    - type: nauc_mrr_at_1000_max
      value: 39.811442869557354
    - type: nauc_mrr_at_1000_std
      value: 23.130168597259633
    - type: nauc_mrr_at_100_diff1
      value: 39.60800937804102
    - type: nauc_mrr_at_100_max
      value: 39.81499208988893
    - type: nauc_mrr_at_100_std
      value: 23.157626527293328
    - type: nauc_mrr_at_10_diff1
      value: 39.699195043308514
    - type: nauc_mrr_at_10_max
      value: 40.135379961488255
    - type: nauc_mrr_at_10_std
      value: 23.20287097378104
    - type: nauc_mrr_at_1_diff1
      value: 41.61660765429233
    - type: nauc_mrr_at_1_max
      value: 34.296676392780164
    - type: nauc_mrr_at_1_std
      value: 18.32511578460585
    - type: nauc_mrr_at_20_diff1
      value: 39.56543789508273
    - type: nauc_mrr_at_20_max
      value: 39.912772847130746
    - type: nauc_mrr_at_20_std
      value: 23.220799720122965
    - type: nauc_mrr_at_3_diff1
      value: 41.558644626992475
    - type: nauc_mrr_at_3_max
      value: 38.66702619348688
    - type: nauc_mrr_at_3_std
      value: 20.819707427206062
    - type: nauc_mrr_at_5_diff1
      value: 40.61298765750478
    - type: nauc_mrr_at_5_max
      value: 39.5367624676024
    - type: nauc_mrr_at_5_std
      value: 22.417103243640522
    - type: nauc_ndcg_at_1000_diff1
      value: 32.41669085796749
    - type: nauc_ndcg_at_1000_max
      value: 39.3889592555104
    - type: nauc_ndcg_at_1000_std
      value: 24.29677240633455
    - type: nauc_ndcg_at_100_diff1
      value: 33.242790506882834
    - type: nauc_ndcg_at_100_max
      value: 33.898859819559526
    - type: nauc_ndcg_at_100_std
      value: 18.61550812422668
    - type: nauc_ndcg_at_10_diff1
      value: 30.975196813400373
    - type: nauc_ndcg_at_10_max
      value: 36.603905951044055
    - type: nauc_ndcg_at_10_std
      value: 22.910262023142636
    - type: nauc_ndcg_at_1_diff1
      value: 40.93200183313147
    - type: nauc_ndcg_at_1_max
      value: 33.727327384812405
    - type: nauc_ndcg_at_1_std
      value: 17.786878916105632
    - type: nauc_ndcg_at_20_diff1
      value: 30.73076927291107
    - type: nauc_ndcg_at_20_max
      value: 35.390532544063156
    - type: nauc_ndcg_at_20_std
      value: 23.078635598419257
    - type: nauc_ndcg_at_3_diff1
      value: 33.658740738364365
    - type: nauc_ndcg_at_3_max
      value: 36.13180757709057
    - type: nauc_ndcg_at_3_std
      value: 19.98271369295018
    - type: nauc_ndcg_at_5_diff1
      value: 32.84341462025322
    - type: nauc_ndcg_at_5_max
      value: 36.28085467882011
    - type: nauc_ndcg_at_5_std
      value: 20.68800320297932
    - type: nauc_precision_at_1000_diff1
      value: -6.708168699596635
    - type: nauc_precision_at_1000_max
      value: 21.89155039384331
    - type: nauc_precision_at_1000_std
      value: 41.62103756428509
    - type: nauc_precision_at_100_diff1
      value: -4.2462696352518785
    - type: nauc_precision_at_100_max
      value: 28.116955819732453
    - type: nauc_precision_at_100_std
      value: 41.3128873672253
    - type: nauc_precision_at_10_diff1
      value: 13.94389231812758
    - type: nauc_precision_at_10_max
      value: 40.16753554948771
    - type: nauc_precision_at_10_std
      value: 34.808485925255724
    - type: nauc_precision_at_1_diff1
      value: 41.61660765429233
    - type: nauc_precision_at_1_max
      value: 34.296676392780164
    - type: nauc_precision_at_1_std
      value: 18.32511578460585
    - type: nauc_precision_at_20_diff1
      value: 6.987363832906253
    - type: nauc_precision_at_20_max
      value: 36.40277188883766
    - type: nauc_precision_at_20_std
      value: 37.97660622154215
    - type: nauc_precision_at_3_diff1
      value: 26.51797121166522
    - type: nauc_precision_at_3_max
      value: 38.0225037150749
    - type: nauc_precision_at_3_std
      value: 23.75378167439284
    - type: nauc_precision_at_5_diff1
      value: 22.753969907185443
    - type: nauc_precision_at_5_max
      value: 38.65374254523585
    - type: nauc_precision_at_5_std
      value: 26.53319799305755
    - type: nauc_recall_at_1000_diff1
      value: 10.443813741265865
    - type: nauc_recall_at_1000_max
      value: 16.714137409220196
    - type: nauc_recall_at_1000_std
      value: 13.22317774140933
    - type: nauc_recall_at_100_diff1
      value: 19.626335453220104
    - type: nauc_recall_at_100_max
      value: 16.781431613595686
    - type: nauc_recall_at_100_std
      value: 4.407503051751801
    - type: nauc_recall_at_10_diff1
      value: 34.40744658686842
    - type: nauc_recall_at_10_max
      value: 19.6970697427996
    - type: nauc_recall_at_10_std
      value: 0.06436545050773496
    - type: nauc_recall_at_1_diff1
      value: 50.61355369486035
    - type: nauc_recall_at_1_max
      value: 7.072899231939653
    - type: nauc_recall_at_1_std
      value: -8.829084633669
    - type: nauc_recall_at_20_diff1
      value: 30.36339862343048
    - type: nauc_recall_at_20_max
      value: 21.46180231299773
    - type: nauc_recall_at_20_std
      value: 2.5214200672465066
    - type: nauc_recall_at_3_diff1
      value: 43.344473400499524
    - type: nauc_recall_at_3_max
      value: 15.27252008740136
    - type: nauc_recall_at_3_std
      value: -5.759955800270036
    - type: nauc_recall_at_5_diff1
      value: 40.3140576520481
    - type: nauc_recall_at_5_max
      value: 18.203301814299262
    - type: nauc_recall_at_5_std
      value: -2.833863969265613
    - type: ndcg_at_1
      value: 36.068
    - type: ndcg_at_10
      value: 27.16
    - type: ndcg_at_100
      value: 25.111
    - type: ndcg_at_1000
      value: 33.936
    - type: ndcg_at_20
      value: 25.572
    - type: ndcg_at_3
      value: 31.657999999999998
    - type: ndcg_at_5
      value: 29.409999999999997
    - type: precision_at_1
      value: 38.080000000000005
    - type: precision_at_10
      value: 19.721
    - type: precision_at_100
      value: 6.494999999999999
    - type: precision_at_1000
      value: 1.9009999999999998
    - type: precision_at_20
      value: 15.262999999999998
    - type: precision_at_3
      value: 29.412
    - type: precision_at_5
      value: 24.892
    - type: recall_at_1
      value: 4.561
    - type: recall_at_10
      value: 13.171
    - type: recall_at_100
      value: 26.686
    - type: recall_at_1000
      value: 58.370999999999995
    - type: recall_at_20
      value: 16.256
    - type: recall_at_3
      value: 8.121
    - type: recall_at_5
      value: 10.015
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB NQ (default)
      revision: b774495ed302d8c44a3a7ea25c90dbce03968f31
      split: test
      type: mteb/nq
    metrics:
    - type: main_score
      value: 21.157999999999998
    - type: map_at_1
      value: 8.819
    - type: map_at_10
      value: 16.304
    - type: map_at_100
      value: 17.589
    - type: map_at_1000
      value: 17.686
    - type: map_at_20
      value: 16.996
    - type: map_at_3
      value: 13.623
    - type: map_at_5
      value: 15.052999999999999
    - type: mrr_at_1
      value: 10.168018539976826
    - type: mrr_at_10
      value: 18.060050304401404
    - type: mrr_at_100
      value: 19.238744437400303
    - type: mrr_at_1000
      value: 19.319259507478357
    - type: mrr_at_20
      value: 18.714121979500284
    - type: mrr_at_3
      value: 15.358246427191933
    - type: mrr_at_5
      value: 16.895036693704107
    - type: nauc_map_at_1000_diff1
      value: 16.229939570708336
    - type: nauc_map_at_1000_max
      value: 12.51276425335657
    - type: nauc_map_at_1000_std
      value: 0.017921811151751794
    - type: nauc_map_at_100_diff1
      value: 16.224170935226457
    - type: nauc_map_at_100_max
      value: 12.512374428428034
    - type: nauc_map_at_100_std
      value: -0.014129877859965362
    - type: nauc_map_at_10_diff1
      value: 16.361123231572876
    - type: nauc_map_at_10_max
      value: 12.052286808517218
    - type: nauc_map_at_10_std
      value: -1.0845583297624024
    - type: nauc_map_at_1_diff1
      value: 21.69479889364372
    - type: nauc_map_at_1_max
      value: 8.751825885396515
    - type: nauc_map_at_1_std
      value: -4.141691109841626
    - type: nauc_map_at_20_diff1
      value: 16.163837857786525
    - type: nauc_map_at_20_max
      value: 12.169273900826845
    - type: nauc_map_at_20_std
      value: -0.5230657088878216
    - type: nauc_map_at_3_diff1
      value: 17.00414872130665
    - type: nauc_map_at_3_max
      value: 10.183926939188352
    - type: nauc_map_at_3_std
      value: -2.7027904518930153
    - type: nauc_map_at_5_diff1
      value: 16.753072970093537
    - type: nauc_map_at_5_max
      value: 11.053795082504381
    - type: nauc_map_at_5_std
      value: -1.8553370232908792
    - type: nauc_mrr_at_1000_diff1
      value: 15.193441578243188
    - type: nauc_mrr_at_1000_max
      value: 11.723486184894917
    - type: nauc_mrr_at_1000_std
      value: 1.1268841139934582
    - type: nauc_mrr_at_100_diff1
      value: 15.181630298990894
    - type: nauc_mrr_at_100_max
      value: 11.726527351488125
    - type: nauc_mrr_at_100_std
      value: 1.1184409088822183
    - type: nauc_mrr_at_10_diff1
      value: 15.234123726061952
    - type: nauc_mrr_at_10_max
      value: 11.362873518294107
    - type: nauc_mrr_at_10_std
      value: 0.3280013067153407
    - type: nauc_mrr_at_1_diff1
      value: 20.323504485666053
    - type: nauc_mrr_at_1_max
      value: 8.409582172937483
    - type: nauc_mrr_at_1_std
      value: -1.7878562506897404
    - type: nauc_mrr_at_20_diff1
      value: 15.140222590320127
    - type: nauc_mrr_at_20_max
      value: 11.507085338033452
    - type: nauc_mrr_at_20_std
      value: 0.7869627376407451
    - type: nauc_mrr_at_3_diff1
      value: 15.787762647299598
    - type: nauc_mrr_at_3_max
      value: 9.735742984997959
    - type: nauc_mrr_at_3_std
      value: -0.7960248100230818
    - type: nauc_mrr_at_5_diff1
      value: 15.457910256551852
    - type: nauc_mrr_at_5_max
      value: 10.453812095004862
    - type: nauc_mrr_at_5_std
      value: -0.19761864109516053
    - type: nauc_ndcg_at_1000_diff1
      value: 14.525260995964645
    - type: nauc_ndcg_at_1000_max
      value: 16.40422064372703
    - type: nauc_ndcg_at_1000_std
      value: 6.160846861359799
    - type: nauc_ndcg_at_100_diff1
      value: 14.43475714612311
    - type: nauc_ndcg_at_100_max
      value: 16.126427718957963
    - type: nauc_ndcg_at_100_std
      value: 5.503813378756256
    - type: nauc_ndcg_at_10_diff1
      value: 14.653040189468122
    - type: nauc_ndcg_at_10_max
      value: 13.702892971289238
    - type: nauc_ndcg_at_10_std
      value: 0.7711212863166211
    - type: nauc_ndcg_at_1_diff1
      value: 20.527077947621663
    - type: nauc_ndcg_at_1_max
      value: 8.484625582388178
    - type: nauc_ndcg_at_1_std
      value: -1.6998416683410011
    - type: nauc_ndcg_at_20_diff1
      value: 14.234859827158017
    - type: nauc_ndcg_at_20_max
      value: 14.11446527457659
    - type: nauc_ndcg_at_20_std
      value: 2.461159703742406
    - type: nauc_ndcg_at_3_diff1
      value: 15.72722174801643
    - type: nauc_ndcg_at_3_max
      value: 10.422853598151251
    - type: nauc_ndcg_at_3_std
      value: -1.8387845418902335
    - type: nauc_ndcg_at_5_diff1
      value: 15.317720430106904
    - type: nauc_ndcg_at_5_max
      value: 11.770806512745684
    - type: nauc_ndcg_at_5_std
      value: -0.5685297839614628
    - type: nauc_precision_at_1000_diff1
      value: -0.01189627743991246
    - type: nauc_precision_at_1000_max
      value: 21.456610537983106
    - type: nauc_precision_at_1000_std
      value: 28.18953073757473
    - type: nauc_precision_at_100_diff1
      value: 5.914625006277201
    - type: nauc_precision_at_100_max
      value: 22.377437220824355
    - type: nauc_precision_at_100_std
      value: 22.22022082888642
    - type: nauc_precision_at_10_diff1
      value: 10.49047701771545
    - type: nauc_precision_at_10_max
      value: 17.080750684938735
    - type: nauc_precision_at_10_std
      value: 6.195671573812195
    - type: nauc_precision_at_1_diff1
      value: 20.527077947621663
    - type: nauc_precision_at_1_max
      value: 8.484625582388178
    - type: nauc_precision_at_1_std
      value: -1.6998416683410011
    - type: nauc_precision_at_20_diff1
      value: 8.838713882019308
    - type: nauc_precision_at_20_max
      value: 17.62766208235687
    - type: nauc_precision_at_20_std
      value: 11.005626718130639
    - type: nauc_precision_at_3_diff1
      value: 12.633919865132393
    - type: nauc_precision_at_3_max
      value: 10.985135077103243
    - type: nauc_precision_at_3_std
      value: 0.669075735470292
    - type: nauc_precision_at_5_diff1
      value: 11.951488524299245
    - type: nauc_precision_at_5_max
      value: 13.48960082923163
    - type: nauc_precision_at_5_std
      value: 2.8417529161489665
    - type: nauc_recall_at_1000_diff1
      value: 9.755913952030559
    - type: nauc_recall_at_1000_max
      value: 42.939720406292956
    - type: nauc_recall_at_1000_std
      value: 44.626368387595214
    - type: nauc_recall_at_100_diff1
      value: 10.72072635332075
    - type: nauc_recall_at_100_max
      value: 27.53559494029257
    - type: nauc_recall_at_100_std
      value: 22.05735513477615
    - type: nauc_recall_at_10_diff1
      value: 11.489494034849928
    - type: nauc_recall_at_10_max
      value: 16.996589598773557
    - type: nauc_recall_at_10_std
      value: 3.4044252519447116
    - type: nauc_recall_at_1_diff1
      value: 21.69479889364372
    - type: nauc_recall_at_1_max
      value: 8.751825885396515
    - type: nauc_recall_at_1_std
      value: -4.141691109841626
    - type: nauc_recall_at_20_diff1
      value: 10.527813840222096
    - type: nauc_recall_at_20_max
      value: 17.992129475663315
    - type: nauc_recall_at_20_std
      value: 7.854098046672443
    - type: nauc_recall_at_3_diff1
      value: 13.388408419008544
    - type: nauc_recall_at_3_max
      value: 11.008383225327554
    - type: nauc_recall_at_3_std
      value: -1.280850893470065
    - type: nauc_recall_at_5_diff1
      value: 12.859671871295616
    - type: nauc_recall_at_5_max
      value: 13.256125187808374
    - type: nauc_recall_at_5_std
      value: 0.9916132686714296
    - type: ndcg_at_1
      value: 10.139
    - type: ndcg_at_10
      value: 21.157999999999998
    - type: ndcg_at_100
      value: 27.668
    - type: ndcg_at_1000
      value: 30.285
    - type: ndcg_at_20
      value: 23.569000000000003
    - type: ndcg_at_3
      value: 15.64
    - type: ndcg_at_5
      value: 18.257
    - type: precision_at_1
      value: 10.139
    - type: precision_at_10
      value: 4.067
    - type: precision_at_100
      value: 0.777
    - type: precision_at_1000
      value: 0.10300000000000001
    - type: precision_at_20
      value: 2.598
    - type: precision_at_3
      value: 7.648000000000001
    - type: precision_at_5
      value: 6.0600000000000005
    - type: recall_at_1
      value: 8.819
    - type: recall_at_10
      value: 34.536
    - type: recall_at_100
      value: 64.781
    - type: recall_at_1000
      value: 84.859
    - type: recall_at_20
      value: 43.559
    - type: recall_at_3
      value: 19.783
    - type: recall_at_5
      value: 25.966
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB QuoraRetrieval (default)
      revision: e4e08e0b7dbe3c8700f0daef558ff32256715259
      split: test
      type: mteb/quora
    metrics:
    - type: main_score
      value: 82.012
    - type: map_at_1
      value: 64.592
    - type: map_at_10
      value: 77.402
    - type: map_at_100
      value: 78.148
    - type: map_at_1000
      value: 78.18
    - type: map_at_20
      value: 77.879
    - type: map_at_3
      value: 74.311
    - type: map_at_5
      value: 76.20400000000001
    - type: mrr_at_1
      value: 74.2
    - type: mrr_at_10
      value: 81.38854365079324
    - type: mrr_at_100
      value: 81.59512445792092
    - type: mrr_at_1000
      value: 81.60023780325119
    - type: mrr_at_20
      value: 81.5363798885578
    - type: mrr_at_3
      value: 79.95333333333294
    - type: mrr_at_5
      value: 80.87783333333273
    - type: nauc_map_at_1000_diff1
      value: 71.563144427743
    - type: nauc_map_at_1000_max
      value: 36.7381066072989
    - type: nauc_map_at_1000_std
      value: -21.980738033181833
    - type: nauc_map_at_100_diff1
      value: 71.56530276635175
    - type: nauc_map_at_100_max
      value: 36.72147688067881
    - type: nauc_map_at_100_std
      value: -22.01274484366413
    - type: nauc_map_at_10_diff1
      value: 71.54359439771537
    - type: nauc_map_at_10_max
      value: 36.26354669846701
    - type: nauc_map_at_10_std
      value: -23.013337979083907
    - type: nauc_map_at_1_diff1
      value: 73.80903272269792
    - type: nauc_map_at_1_max
      value: 29.97715650504089
    - type: nauc_map_at_1_std
      value: -21.00007771976864
    - type: nauc_map_at_20_diff1
      value: 71.57688339120256
    - type: nauc_map_at_20_max
      value: 36.618589942433395
    - type: nauc_map_at_20_std
      value: -22.33064443082938
    - type: nauc_map_at_3_diff1
      value: 71.72670742525665
    - type: nauc_map_at_3_max
      value: 34.6281120452265
    - type: nauc_map_at_3_std
      value: -23.899175833520125
    - type: nauc_map_at_5_diff1
      value: 71.47259917882022
    - type: nauc_map_at_5_max
      value: 35.62412436441212
    - type: nauc_map_at_5_std
      value: -23.47998884289205
    - type: nauc_mrr_at_1000_diff1
      value: 72.80796546579286
    - type: nauc_mrr_at_1000_max
      value: 39.25693300384676
    - type: nauc_mrr_at_1000_std
      value: -19.601139084660538
    - type: nauc_mrr_at_100_diff1
      value: 72.80693590878819
    - type: nauc_mrr_at_100_max
      value: 39.25913251601793
    - type: nauc_mrr_at_100_std
      value: -19.597248510226006
    - type: nauc_mrr_at_10_diff1
      value: 72.73505807995161
    - type: nauc_mrr_at_10_max
      value: 39.276454616539716
    - type: nauc_mrr_at_10_std
      value: -19.717754697854193
    - type: nauc_mrr_at_1_diff1
      value: 74.32519392972422
    - type: nauc_mrr_at_1_max
      value: 38.476636619971174
    - type: nauc_mrr_at_1_std
      value: -19.079420393939518
    - type: nauc_mrr_at_20_diff1
      value: 72.78854375041742
    - type: nauc_mrr_at_20_max
      value: 39.31033803827273
    - type: nauc_mrr_at_20_std
      value: -19.55603905094717
    - type: nauc_mrr_at_3_diff1
      value: 72.45226928418506
    - type: nauc_mrr_at_3_max
      value: 39.02048177756944
    - type: nauc_mrr_at_3_std
      value: -19.990652853472366
    - type: nauc_mrr_at_5_diff1
      value: 72.55978195949999
    - type: nauc_mrr_at_5_max
      value: 39.23684364430699
    - type: nauc_mrr_at_5_std
      value: -19.821588662337117
    - type: nauc_ndcg_at_1000_diff1
      value: 71.53703986610873
    - type: nauc_ndcg_at_1000_max
      value: 38.2390814708584
    - type: nauc_ndcg_at_1000_std
      value: -20.26092896395758
    - type: nauc_ndcg_at_100_diff1
      value: 71.53867175889835
    - type: nauc_ndcg_at_100_max
      value: 38.13196515267871
    - type: nauc_ndcg_at_100_std
      value: -20.251462418761683
    - type: nauc_ndcg_at_10_diff1
      value: 71.0650335459164
    - type: nauc_ndcg_at_10_max
      value: 37.38571549486721
    - type: nauc_ndcg_at_10_std
      value: -22.258604100812555
    - type: nauc_ndcg_at_1_diff1
      value: 74.22381616523879
    - type: nauc_ndcg_at_1_max
      value: 38.678662109624455
    - type: nauc_ndcg_at_1_std
      value: -18.93974180166645
    - type: nauc_ndcg_at_20_diff1
      value: 71.42183736397989
    - type: nauc_ndcg_at_20_max
      value: 38.04822708035827
    - type: nauc_ndcg_at_20_std
      value: -20.868258633569194
    - type: nauc_ndcg_at_3_diff1
      value: 70.56404049371294
    - type: nauc_ndcg_at_3_max
      value: 36.44618012763879
    - type: nauc_ndcg_at_3_std
      value: -22.537077598994543
    - type: nauc_ndcg_at_5_diff1
      value: 70.59978302123125
    - type: nauc_ndcg_at_5_max
      value: 36.83084353944159
    - type: nauc_ndcg_at_5_std
      value: -22.506208564791557
    - type: nauc_precision_at_1000_diff1
      value: -37.04415583634801
    - type: nauc_precision_at_1000_max
      value: -5.065719458200828
    - type: nauc_precision_at_1000_std
      value: 22.616085445440955
    - type: nauc_precision_at_100_diff1
      value: -34.82554531006301
    - type: nauc_precision_at_100_max
      value: -3.0424194261578172
    - type: nauc_precision_at_100_std
      value: 21.684667518110853
    - type: nauc_precision_at_10_diff1
      value: -23.077641035092185
    - type: nauc_precision_at_10_max
      value: 5.36093469591455
    - type: nauc_precision_at_10_std
      value: 11.906515810765315
    - type: nauc_precision_at_1_diff1
      value: 74.22381616523879
    - type: nauc_precision_at_1_max
      value: 38.678662109624455
    - type: nauc_precision_at_1_std
      value: -18.93974180166645
    - type: nauc_precision_at_20_diff1
      value: -29.103864491887173
    - type: nauc_precision_at_20_max
      value: 1.9096690929564353
    - type: nauc_precision_at_20_std
      value: 17.636958366277263
    - type: nauc_precision_at_3_diff1
      value: 4.397256394687356
    - type: nauc_precision_at_3_max
      value: 17.01622079466081
    - type: nauc_precision_at_3_std
      value: -1.998336590418495
    - type: nauc_precision_at_5_diff1
      value: -10.944029853590333
    - type: nauc_precision_at_5_max
      value: 11.352086411823322
    - type: nauc_precision_at_5_std
      value: 5.5348305497123755
    - type: nauc_recall_at_1000_diff1
      value: 57.63858758830721
    - type: nauc_recall_at_1000_max
      value: 56.154820307155084
    - type: nauc_recall_at_1000_std
      value: 37.297558664297455
    - type: nauc_recall_at_100_diff1
      value: 65.12971448538715
    - type: nauc_recall_at_100_max
      value: 37.459434024112326
    - type: nauc_recall_at_100_std
      value: -3.4712878121583337
    - type: nauc_recall_at_10_diff1
      value: 64.05584648206661
    - type: nauc_recall_at_10_max
      value: 34.334390035941865
    - type: nauc_recall_at_10_std
      value: -26.024428985438554
    - type: nauc_recall_at_1_diff1
      value: 73.80903272269792
    - type: nauc_recall_at_1_max
      value: 29.97715650504089
    - type: nauc_recall_at_1_std
      value: -21.00007771976864
    - type: nauc_recall_at_20_diff1
      value: 64.81380914632466
    - type: nauc_recall_at_20_max
      value: 38.06950127045061
    - type: nauc_recall_at_20_std
      value: -15.95444205912522
    - type: nauc_recall_at_3_diff1
      value: 66.90914558056423
    - type: nauc_recall_at_3_max
      value: 32.23359446748229
    - type: nauc_recall_at_3_std
      value: -26.18188347743054
    - type: nauc_recall_at_5_diff1
      value: 65.0856724732645
    - type: nauc_recall_at_5_max
      value: 33.26433283889179
    - type: nauc_recall_at_5_std
      value: -26.59355203187721
    - type: ndcg_at_1
      value: 74.25
    - type: ndcg_at_10
      value: 82.012
    - type: ndcg_at_100
      value: 83.907
    - type: ndcg_at_1000
      value: 84.237
    - type: ndcg_at_20
      value: 82.98299999999999
    - type: ndcg_at_3
      value: 78.318
    - type: ndcg_at_5
      value: 80.257
    - type: precision_at_1
      value: 74.25
    - type: precision_at_10
      value: 12.433
    - type: precision_at_100
      value: 1.471
    - type: precision_at_1000
      value: 0.155
    - type: precision_at_20
      value: 6.679
    - type: precision_at_3
      value: 33.947
    - type: precision_at_5
      value: 22.52
    - type: recall_at_1
      value: 64.592
    - type: recall_at_10
      value: 90.99000000000001
    - type: recall_at_100
      value: 97.878
    - type: recall_at_1000
      value: 99.69
    - type: recall_at_20
      value: 94.21000000000001
    - type: recall_at_3
      value: 80.487
    - type: recall_at_5
      value: 85.794
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB RedditClustering (default)
      revision: 24640382cdbf8abc73003fb0fa6d111a705499eb
      split: test
      type: mteb/reddit-clustering
    metrics:
    - type: main_score
      value: 43.25840276111764
    - type: v_measure
      value: 43.25840276111764
    - type: v_measure_std
      value: 5.550064583648558
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB RedditClusteringP2P (default)
      revision: 385e3cb46b4cfa89021f56c4380204149d0efe33
      split: test
      type: mteb/reddit-clustering-p2p
    metrics:
    - type: main_score
      value: 49.64164121828545
    - type: v_measure
      value: 49.64164121828545
    - type: v_measure_std
      value: 10.85687862239164
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB SCIDOCS (default)
      revision: f8c2fcf00f625baaa80f62ec5bd9e1fff3b8ae88
      split: test
      type: mteb/scidocs
    metrics:
    - type: main_score
      value: 13.587
    - type: map_at_1
      value: 3.3680000000000003
    - type: map_at_10
      value: 7.712
    - type: map_at_100
      value: 9.141
    - type: map_at_1000
      value: 9.414
    - type: map_at_20
      value: 8.419
    - type: map_at_3
      value: 5.683
    - type: map_at_5
      value: 6.627
    - type: mrr_at_1
      value: 16.6
    - type: mrr_at_10
      value: 24.389166666666657
    - type: mrr_at_100
      value: 25.564348279710202
    - type: mrr_at_1000
      value: 25.65219733433818
    - type: mrr_at_20
      value: 25.011801694607428
    - type: mrr_at_3
      value: 21.50000000000001
    - type: mrr_at_5
      value: 23.075000000000006
    - type: nauc_map_at_1000_diff1
      value: 15.792054559873389
    - type: nauc_map_at_1000_max
      value: 13.56634783353828
    - type: nauc_map_at_1000_std
      value: 13.522494963276946
    - type: nauc_map_at_100_diff1
      value: 15.75666330397116
    - type: nauc_map_at_100_max
      value: 13.371620548211776
    - type: nauc_map_at_100_std
      value: 13.041945684128557
    - type: nauc_map_at_10_diff1
      value: 15.948702753831338
    - type: nauc_map_at_10_max
      value: 11.707911515178242
    - type: nauc_map_at_10_std
      value: 8.727495324693498
    - type: nauc_map_at_1_diff1
      value: 17.29360255890909
    - type: nauc_map_at_1_max
      value: 10.72310668533521
    - type: nauc_map_at_1_std
      value: 4.112759851902581
    - type: nauc_map_at_20_diff1
      value: 16.037701154664326
    - type: nauc_map_at_20_max
      value: 12.828849813839474
    - type: nauc_map_at_20_std
      value: 10.795705865130165
    - type: nauc_map_at_3_diff1
      value: 18.23195167071921
    - type: nauc_map_at_3_max
      value: 9.801116024411058
    - type: nauc_map_at_3_std
      value: 5.213983881776522
    - type: nauc_map_at_5_diff1
      value: 17.5626604163625
    - type: nauc_map_at_5_max
      value: 9.970369227992304
    - type: nauc_map_at_5_std
      value: 6.735083895176833
    - type: nauc_mrr_at_1000_diff1
      value: 15.510099983807086
    - type: nauc_mrr_at_1000_max
      value: 11.343645637837431
    - type: nauc_mrr_at_1000_std
      value: 7.566937985959651
    - type: nauc_mrr_at_100_diff1
      value: 15.506985809378149
    - type: nauc_mrr_at_100_max
      value: 11.340883862596678
    - type: nauc_mrr_at_100_std
      value: 7.620646271034095
    - type: nauc_mrr_at_10_diff1
      value: 15.490875842909002
    - type: nauc_mrr_at_10_max
      value: 11.231094796251513
    - type: nauc_mrr_at_10_std
      value: 7.130502639986691
    - type: nauc_mrr_at_1_diff1
      value: 16.883526727193278
    - type: nauc_mrr_at_1_max
      value: 10.173624986461805
    - type: nauc_mrr_at_1_std
      value: 4.256034299244473
    - type: nauc_mrr_at_20_diff1
      value: 15.485170013873923
    - type: nauc_mrr_at_20_max
      value: 11.233104363368977
    - type: nauc_mrr_at_20_std
      value: 7.45451688565917
    - type: nauc_mrr_at_3_diff1
      value: 15.34991113239244
    - type: nauc_mrr_at_3_max
      value: 9.961720650018536
    - type: nauc_mrr_at_3_std
      value: 5.38229302774779
    - type: nauc_mrr_at_5_diff1
      value: 15.873465265800968
    - type: nauc_mrr_at_5_max
      value: 10.838561422578016
    - type: nauc_mrr_at_5_std
      value: 6.441461173575752
    - type: nauc_ndcg_at_1000_diff1
      value: 14.352637695984038
    - type: nauc_ndcg_at_1000_max
      value: 17.013971425841103
    - type: nauc_ndcg_at_1000_std
      value: 21.97010413330881
    - type: nauc_ndcg_at_100_diff1
      value: 13.886964830324594
    - type: nauc_ndcg_at_100_max
      value: 15.552160909918156
    - type: nauc_ndcg_at_100_std
      value: 19.26622209227307
    - type: nauc_ndcg_at_10_diff1
      value: 14.639378627164511
    - type: nauc_ndcg_at_10_max
      value: 12.825984197211927
    - type: nauc_ndcg_at_10_std
      value: 10.391285942640947
    - type: nauc_ndcg_at_1_diff1
      value: 16.883526727193278
    - type: nauc_ndcg_at_1_max
      value: 10.173624986461805
    - type: nauc_ndcg_at_1_std
      value: 4.256034299244473
    - type: nauc_ndcg_at_20_diff1
      value: 14.995243704619316
    - type: nauc_ndcg_at_20_max
      value: 14.26525727222302
    - type: nauc_ndcg_at_20_std
      value: 13.639982949452376
    - type: nauc_ndcg_at_3_diff1
      value: 16.91652612412296
    - type: nauc_ndcg_at_3_max
      value: 9.78989931219063
    - type: nauc_ndcg_at_3_std
      value: 5.732330748163062
    - type: nauc_ndcg_at_5_diff1
      value: 17.00999871755564
    - type: nauc_ndcg_at_5_max
      value: 10.924554915253575
    - type: nauc_ndcg_at_5_std
      value: 7.537589771075501
    - type: nauc_precision_at_1000_diff1
      value: 8.009592365347004
    - type: nauc_precision_at_1000_max
      value: 20.05874444978994
    - type: nauc_precision_at_1000_std
      value: 36.79899155088703
    - type: nauc_precision_at_100_diff1
      value: 8.255276158647746
    - type: nauc_precision_at_100_max
      value: 17.580852368830026
    - type: nauc_precision_at_100_std
      value: 30.633816908045635
    - type: nauc_precision_at_10_diff1
      value: 11.553031122695996
    - type: nauc_precision_at_10_max
      value: 14.482109658676057
    - type: nauc_precision_at_10_std
      value: 14.259782606679593
    - type: nauc_precision_at_1_diff1
      value: 16.883526727193278
    - type: nauc_precision_at_1_max
      value: 10.173624986461805
    - type: nauc_precision_at_1_std
      value: 4.256034299244473
    - type: nauc_precision_at_20_diff1
      value: 12.005709487312277
    - type: nauc_precision_at_20_max
      value: 16.45307012250548
    - type: nauc_precision_at_20_std
      value: 19.760010062385792
    - type: nauc_precision_at_3_diff1
      value: 17.121112206808522
    - type: nauc_precision_at_3_max
      value: 10.282548766937094
    - type: nauc_precision_at_3_std
      value: 6.628229870593426
    - type: nauc_precision_at_5_diff1
      value: 16.639581424506922
    - type: nauc_precision_at_5_max
      value: 11.595216280079104
    - type: nauc_precision_at_5_std
      value: 9.706804262911701
    - type: nauc_recall_at_1000_diff1
      value: 7.944908337255163
    - type: nauc_recall_at_1000_max
      value: 20.888140916483884
    - type: nauc_recall_at_1000_std
      value: 37.43522863327242
    - type: nauc_recall_at_100_diff1
      value: 8.145521632469906
    - type: nauc_recall_at_100_max
      value: 17.816240183909578
    - type: nauc_recall_at_100_std
      value: 30.822040834292043
    - type: nauc_recall_at_10_diff1
      value: 11.626320851379472
    - type: nauc_recall_at_10_max
      value: 14.568333470882427
    - type: nauc_recall_at_10_std
      value: 14.086250979150067
    - type: nauc_recall_at_1_diff1
      value: 17.29360255890909
    - type: nauc_recall_at_1_max
      value: 10.72310668533521
    - type: nauc_recall_at_1_std
      value: 4.112759851902581
    - type: nauc_recall_at_20_diff1
      value: 12.125684404199848
    - type: nauc_recall_at_20_max
      value: 16.596620432069138
    - type: nauc_recall_at_20_std
      value: 19.649717933105602
    - type: nauc_recall_at_3_diff1
      value: 17.332586222101803
    - type: nauc_recall_at_3_max
      value: 10.609602723354438
    - type: nauc_recall_at_3_std
      value: 6.326563940802695
    - type: nauc_recall_at_5_diff1
      value: 16.7930840371413
    - type: nauc_recall_at_5_max
      value: 11.809252198719197
    - type: nauc_recall_at_5_std
      value: 9.44067075649909
    - type: ndcg_at_1
      value: 16.6
    - type: ndcg_at_10
      value: 13.587
    - type: ndcg_at_100
      value: 19.980999999999998
    - type: ndcg_at_1000
      value: 25.484
    - type: ndcg_at_20
      value: 15.742
    - type: ndcg_at_3
      value: 12.859000000000002
    - type: ndcg_at_5
      value: 11.135
    - type: precision_at_1
      value: 16.6
    - type: precision_at_10
      value: 7.01
    - type: precision_at_100
      value: 1.6320000000000001
    - type: precision_at_1000
      value: 0.296
    - type: precision_at_20
      value: 4.755
    - type: precision_at_3
      value: 11.799999999999999
    - type: precision_at_5
      value: 9.6
    - type: recall_at_1
      value: 3.3680000000000003
    - type: recall_at_10
      value: 14.193
    - type: recall_at_100
      value: 33.107
    - type: recall_at_1000
      value: 60.145
    - type: recall_at_20
      value: 19.233
    - type: recall_at_3
      value: 7.163
    - type: recall_at_5
      value: 9.713
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB SICK-R (default)
      revision: 20a6d6f312dd54037fe07a32d58e5e168867909d
      split: test
      type: mteb/sickr-sts
    metrics:
    - type: cosine_pearson
      value: 76.25915789536036
    - type: cosine_spearman
      value: 65.66845738179555
    - type: euclidean_pearson
      value: 69.85107857850403
    - type: euclidean_spearman
      value: 65.6685173896875
    - type: main_score
      value: 65.66845738179555
    - type: manhattan_pearson
      value: 69.01865715022275
    - type: manhattan_spearman
      value: 65.63874813005013
    - type: pearson
      value: 76.25915789536036
    - type: spearman
      value: 65.66845738179555
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB STS12 (default)
      revision: a0d554a64d88156834ff5ae9920b964011b16384
      split: test
      type: mteb/sts12-sts
    metrics:
    - type: cosine_pearson
      value: 72.40943128011739
    - type: cosine_spearman
      value: 62.72495538860372
    - type: euclidean_pearson
      value: 68.11171624405146
    - type: euclidean_spearman
      value: 62.72485577613837
    - type: main_score
      value: 62.72495538860372
    - type: manhattan_pearson
      value: 64.813988464561
    - type: manhattan_spearman
      value: 60.793210368567216
    - type: pearson
      value: 72.40943128011739
    - type: spearman
      value: 62.72495538860372
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB STS13 (default)
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
      split: test
      type: mteb/sts13-sts
    metrics:
    - type: cosine_pearson
      value: 76.70618047980291
    - type: cosine_spearman
      value: 77.59358804164447
    - type: euclidean_pearson
      value: 77.10766267433688
    - type: euclidean_spearman
      value: 77.59339585903179
    - type: main_score
      value: 77.59358804164447
    - type: manhattan_pearson
      value: 75.578854286063
    - type: manhattan_spearman
      value: 75.9068297217428
    - type: pearson
      value: 76.70618047980291
    - type: spearman
      value: 77.59358804164447
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB STS14 (default)
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
      split: test
      type: mteb/sts14-sts
    metrics:
    - type: cosine_pearson
      value: 76.94540031581289
    - type: cosine_spearman
      value: 72.90346996863444
    - type: euclidean_pearson
      value: 75.32643341198654
    - type: euclidean_spearman
      value: 72.90349626869781
    - type: main_score
      value: 72.90346996863444
    - type: manhattan_pearson
      value: 74.65306991359576
    - type: manhattan_spearman
      value: 72.51411158597628
    - type: pearson
      value: 76.94540031581289
    - type: spearman
      value: 72.90346996863444
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB STS15 (default)
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
      split: test
      type: mteb/sts15-sts
    metrics:
    - type: cosine_pearson
      value: 80.29008555581484
    - type: cosine_spearman
      value: 80.75923364182438
    - type: euclidean_pearson
      value: 80.31136434402754
    - type: euclidean_spearman
      value: 80.75922963811225
    - type: main_score
      value: 80.75923364182438
    - type: manhattan_pearson
      value: 79.32871995817709
    - type: manhattan_spearman
      value: 79.56283194488046
    - type: pearson
      value: 80.29008555581484
    - type: spearman
      value: 80.75923364182438
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB STS16 (default)
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
      split: test
      type: mteb/sts16-sts
    metrics:
    - type: cosine_pearson
      value: 76.19539001140771
    - type: cosine_spearman
      value: 76.85661925812892
    - type: euclidean_pearson
      value: 76.08663935674932
    - type: euclidean_spearman
      value: 76.85661925812892
    - type: main_score
      value: 76.85661925812892
    - type: manhattan_pearson
      value: 74.81988866329016
    - type: manhattan_spearman
      value: 75.50402310757015
    - type: pearson
      value: 76.19539001140771
    - type: spearman
      value: 76.85661925812892
    task:
      type: STS
  - dataset:
      config: nl-en
      name: MTEB STS17 (nl-en)
      revision: faeb762787bd10488a50c8b5be4a3b82e411949c
      split: test
      type: mteb/sts17-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 34.397932428162186
    - type: cosine_spearman
      value: 29.85196707881456
    - type: euclidean_pearson
      value: 34.64394718355606
    - type: euclidean_spearman
      value: 29.85196707881456
    - type: main_score
      value: 29.85196707881456
    - type: manhattan_pearson
      value: 34.16832023178801
    - type: manhattan_spearman
      value: 33.14486169393415
    - type: pearson
      value: 34.397932428162186
    - type: spearman
      value: 29.85196707881456
    task:
      type: STS
  - dataset:
      config: es-en
      name: MTEB STS17 (es-en)
      revision: faeb762787bd10488a50c8b5be4a3b82e411949c
      split: test
      type: mteb/sts17-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 20.30327751081854
    - type: cosine_spearman
      value: 18.674265253799298
    - type: euclidean_pearson
      value: 20.131998188841948
    - type: euclidean_spearman
      value: 18.674265253799298
    - type: main_score
      value: 18.674265253799298
    - type: manhattan_pearson
      value: 18.622067051882603
    - type: manhattan_spearman
      value: 18.620291055648483
    - type: pearson
      value: 20.30327751081854
    - type: spearman
      value: 18.674265253799298
    task:
      type: STS
  - dataset:
      config: en-de
      name: MTEB STS17 (en-de)
      revision: faeb762787bd10488a50c8b5be4a3b82e411949c
      split: test
      type: mteb/sts17-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 28.49639097934199
    - type: cosine_spearman
      value: 26.76895298058133
    - type: euclidean_pearson
      value: 28.51993606461124
    - type: euclidean_spearman
      value: 26.76895298058133
    - type: main_score
      value: 26.76895298058133
    - type: manhattan_pearson
      value: 28.34674577371768
    - type: manhattan_spearman
      value: 24.811029147686337
    - type: pearson
      value: 28.49639097934199
    - type: spearman
      value: 26.76895298058133
    task:
      type: STS
  - dataset:
      config: en-en
      name: MTEB STS17 (en-en)
      revision: faeb762787bd10488a50c8b5be4a3b82e411949c
      split: test
      type: mteb/sts17-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 85.42068849087106
    - type: cosine_spearman
      value: 87.05993755101053
    - type: euclidean_pearson
      value: 85.50658100259913
    - type: euclidean_spearman
      value: 87.05993755101053
    - type: main_score
      value: 87.05993755101053
    - type: manhattan_pearson
      value: 85.05037515486939
    - type: manhattan_spearman
      value: 86.78286451699647
    - type: pearson
      value: 85.42068849087106
    - type: spearman
      value: 87.05993755101053
    task:
      type: STS
  - dataset:
      config: it-en
      name: MTEB STS17 (it-en)
      revision: faeb762787bd10488a50c8b5be4a3b82e411949c
      split: test
      type: mteb/sts17-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 28.331000764771648
    - type: cosine_spearman
      value: 26.924748902731583
    - type: euclidean_pearson
      value: 27.908262381916842
    - type: euclidean_spearman
      value: 26.924748902731583
    - type: main_score
      value: 26.924748902731583
    - type: manhattan_pearson
      value: 27.64928698735386
    - type: manhattan_spearman
      value: 26.33489239510866
    - type: pearson
      value: 28.331000764771648
    - type: spearman
      value: 26.924748902731583
    task:
      type: STS
  - dataset:
      config: fr-en
      name: MTEB STS17 (fr-en)
      revision: faeb762787bd10488a50c8b5be4a3b82e411949c
      split: test
      type: mteb/sts17-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 35.99128331010763
    - type: cosine_spearman
      value: 33.882787753030755
    - type: euclidean_pearson
      value: 36.125565540276824
    - type: euclidean_spearman
      value: 33.882787753030755
    - type: main_score
      value: 33.882787753030755
    - type: manhattan_pearson
      value: 39.43371979888863
    - type: manhattan_spearman
      value: 39.98846569097863
    - type: pearson
      value: 35.99128331010763
    - type: spearman
      value: 33.882787753030755
    task:
      type: STS
  - dataset:
      config: en-ar
      name: MTEB STS17 (en-ar)
      revision: faeb762787bd10488a50c8b5be4a3b82e411949c
      split: test
      type: mteb/sts17-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 1.3231798311649463
    - type: cosine_spearman
      value: 1.968045578690376
    - type: euclidean_pearson
      value: 1.2443039427500642
    - type: euclidean_spearman
      value: 1.968045578690376
    - type: main_score
      value: 1.968045578690376
    - type: manhattan_pearson
      value: 0.29924785068227155
    - type: manhattan_spearman
      value: 3.1701763139219117
    - type: pearson
      value: 1.3231798311649463
    - type: spearman
      value: 1.968045578690376
    task:
      type: STS
  - dataset:
      config: en-tr
      name: MTEB STS17 (en-tr)
      revision: faeb762787bd10488a50c8b5be4a3b82e411949c
      split: test
      type: mteb/sts17-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 7.571886460510519
    - type: cosine_spearman
      value: 6.426199138705195
    - type: euclidean_pearson
      value: 7.436947146723141
    - type: euclidean_spearman
      value: 6.426199138705195
    - type: main_score
      value: 6.426199138705195
    - type: manhattan_pearson
      value: 5.225717518299594
    - type: manhattan_spearman
      value: 3.067077550441944
    - type: pearson
      value: 7.571886460510519
    - type: spearman
      value: 6.426199138705195
    task:
      type: STS
  - dataset:
      config: en
      name: MTEB STS22 (en)
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 61.088670238491204
    - type: cosine_spearman
      value: 64.25828026896876
    - type: euclidean_pearson
      value: 63.7609187555772
    - type: euclidean_spearman
      value: 64.25828026896876
    - type: main_score
      value: 64.25828026896876
    - type: manhattan_pearson
      value: 62.601398537174035
    - type: manhattan_spearman
      value: 62.87332671301306
    - type: pearson
      value: 61.088670238491204
    - type: spearman
      value: 64.25828026896876
    task:
      type: STS
  - dataset:
      config: de-en
      name: MTEB STS22 (de-en)
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 31.0255615174742
    - type: cosine_spearman
      value: 36.69481137227967
    - type: euclidean_pearson
      value: 29.900021582547264
    - type: euclidean_spearman
      value: 36.69481137227967
    - type: main_score
      value: 36.69481137227967
    - type: manhattan_pearson
      value: 29.619780557503155
    - type: manhattan_spearman
      value: 41.91843653096047
    - type: pearson
      value: 31.0255615174742
    - type: spearman
      value: 36.69481137227967
    task:
      type: STS
  - dataset:
      config: pl-en
      name: MTEB STS22 (pl-en)
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 10.40841334454418
    - type: cosine_spearman
      value: 28.104310738121512
    - type: euclidean_pearson
      value: 8.128707220686382
    - type: euclidean_spearman
      value: 28.104310738121512
    - type: main_score
      value: 28.104310738121512
    - type: manhattan_pearson
      value: 14.726925529355325
    - type: manhattan_spearman
      value: 28.057426809179326
    - type: pearson
      value: 10.40841334454418
    - type: spearman
      value: 28.104310738121512
    task:
      type: STS
  - dataset:
      config: es-en
      name: MTEB STS22 (es-en)
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 10.755722104172923
    - type: cosine_spearman
      value: 12.444878649130276
    - type: euclidean_pearson
      value: 10.059477626348706
    - type: euclidean_spearman
      value: 12.468003510617354
    - type: main_score
      value: 12.444878649130276
    - type: manhattan_pearson
      value: 13.507609577564672
    - type: manhattan_spearman
      value: 18.390599199214037
    - type: pearson
      value: 10.755722104172923
    - type: spearman
      value: 12.444878649130276
    task:
      type: STS
  - dataset:
      config: zh-en
      name: MTEB STS22 (zh-en)
      revision: de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3
      split: test
      type: mteb/sts22-crosslingual-sts
    metrics:
    - type: cosine_pearson
      value: 13.26815146152796
    - type: cosine_spearman
      value: 18.433007526015338
    - type: euclidean_pearson
      value: 12.020944164266574
    - type: euclidean_spearman
      value: 18.426334503093322
    - type: main_score
      value: 18.433007526015338
    - type: manhattan_pearson
      value: 11.933248448259237
    - type: manhattan_spearman
      value: 18.324625546075126
    - type: pearson
      value: 13.26815146152796
    - type: spearman
      value: 18.433007526015338
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB STSBenchmark (default)
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
      split: test
      type: mteb/stsbenchmark-sts
    metrics:
    - type: cosine_pearson
      value: 78.68909918763245
    - type: cosine_spearman
      value: 76.83852449552488
    - type: euclidean_pearson
      value: 78.35108573675653
    - type: euclidean_spearman
      value: 76.83846447877973
    - type: main_score
      value: 76.83852449552488
    - type: manhattan_pearson
      value: 77.02876135749386
    - type: manhattan_spearman
      value: 75.9170365531251
    - type: pearson
      value: 78.68909918763245
    - type: spearman
      value: 76.83852449552488
    task:
      type: STS
  - dataset:
      config: default
      name: MTEB SciDocsRR (default)
      revision: d3c5e1fc0b855ab6097bf1cda04dd73947d7caab
      split: test
      type: mteb/scidocs-reranking
    metrics:
    - type: main_score
      value: 74.82680371550936
    - type: map
      value: 74.82680371550936
    - type: mrr
      value: 91.93418963026807
    - type: nAUC_map_diff1
      value: 9.068255025841692
    - type: nAUC_map_max
      value: 53.83045488537594
    - type: nAUC_map_std
      value: 65.66457388284374
    - type: nAUC_mrr_diff1
      value: 46.05085316225451
    - type: nAUC_mrr_max
      value: 75.59509312588396
    - type: nAUC_mrr_std
      value: 70.87824623580906
    task:
      type: Reranking
  - dataset:
      config: default
      name: MTEB SciFact (default)
      revision: 0228b52cf27578f30900b9e5271d331663a030d7
      split: test
      type: mteb/scifact
    metrics:
    - type: main_score
      value: 56.2
    - type: map_at_1
      value: 40.666999999999994
    - type: map_at_10
      value: 51.129999999999995
    - type: map_at_100
      value: 52.056999999999995
    - type: map_at_1000
      value: 52.098
    - type: map_at_20
      value: 51.686
    - type: map_at_3
      value: 48.634
    - type: map_at_5
      value: 50.066
    - type: mrr_at_1
      value: 42.66666666666667
    - type: mrr_at_10
      value: 52.482142857142854
    - type: mrr_at_100
      value: 53.19823245085252
    - type: mrr_at_1000
      value: 53.23357269611506
    - type: mrr_at_20
      value: 52.87664766434116
    - type: mrr_at_3
      value: 50.499999999999986
    - type: mrr_at_5
      value: 51.4
    - type: nauc_map_at_1000_diff1
      value: 55.679524170340144
    - type: nauc_map_at_1000_max
      value: 35.34368225457809
    - type: nauc_map_at_1000_std
      value: -2.043911326695919
    - type: nauc_map_at_100_diff1
      value: 55.64964353245028
    - type: nauc_map_at_100_max
      value: 35.33722631779892
    - type: nauc_map_at_100_std
      value: -2.0539617122568625
    - type: nauc_map_at_10_diff1
      value: 55.82499509356933
    - type: nauc_map_at_10_max
      value: 34.59317781850682
    - type: nauc_map_at_10_std
      value: -3.1860541248309358
    - type: nauc_map_at_1_diff1
      value: 59.10174915043726
    - type: nauc_map_at_1_max
      value: 33.416492729123185
    - type: nauc_map_at_1_std
      value: -5.1952700488175045
    - type: nauc_map_at_20_diff1
      value: 55.623431232446904
    - type: nauc_map_at_20_max
      value: 35.296574838119085
    - type: nauc_map_at_20_std
      value: -2.007783275001443
    - type: nauc_map_at_3_diff1
      value: 56.556766655254485
    - type: nauc_map_at_3_max
      value: 33.3155647194758
    - type: nauc_map_at_3_std
      value: -3.7964967547169106
    - type: nauc_map_at_5_diff1
      value: 56.553418502313676
    - type: nauc_map_at_5_max
      value: 34.13639468025285
    - type: nauc_map_at_5_std
      value: -3.9708550035335346
    - type: nauc_mrr_at_1000_diff1
      value: 55.331555076880576
    - type: nauc_mrr_at_1000_max
      value: 37.027463444459606
    - type: nauc_mrr_at_1000_std
      value: 1.9690692773511222
    - type: nauc_mrr_at_100_diff1
      value: 55.28850823087701
    - type: nauc_mrr_at_100_max
      value: 37.03565595888509
    - type: nauc_mrr_at_100_std
      value: 1.9754212228406436
    - type: nauc_mrr_at_10_diff1
      value: 55.35398010609678
    - type: nauc_mrr_at_10_max
      value: 36.813749789783465
    - type: nauc_mrr_at_10_std
      value: 1.69908309119671
    - type: nauc_mrr_at_1_diff1
      value: 59.698357986628324
    - type: nauc_mrr_at_1_max
      value: 36.19184535975336
    - type: nauc_mrr_at_1_std
      value: -0.3732635881802827
    - type: nauc_mrr_at_20_diff1
      value: 55.26305584448552
    - type: nauc_mrr_at_20_max
      value: 37.11198171015867
    - type: nauc_mrr_at_20_std
      value: 2.1612218864195816
    - type: nauc_mrr_at_3_diff1
      value: 56.25662215357208
    - type: nauc_mrr_at_3_max
      value: 35.8103147848101
    - type: nauc_mrr_at_3_std
      value: 1.408422730019326
    - type: nauc_mrr_at_5_diff1
      value: 56.409279532488
    - type: nauc_mrr_at_5_max
      value: 36.24764657740795
    - type: nauc_mrr_at_5_std
      value: 0.9191877090789675
    - type: nauc_ndcg_at_1000_diff1
      value: 53.921052108891374
    - type: nauc_ndcg_at_1000_max
      value: 37.23673471367524
    - type: nauc_ndcg_at_1000_std
      value: 1.2797242206174546
    - type: nauc_ndcg_at_100_diff1
      value: 53.08910074626929
    - type: nauc_ndcg_at_100_max
      value: 37.58307549599563
    - type: nauc_ndcg_at_100_std
      value: 1.6730489754502
    - type: nauc_ndcg_at_10_diff1
      value: 53.4900294437438
    - type: nauc_ndcg_at_10_max
      value: 35.63914186917353
    - type: nauc_ndcg_at_10_std
      value: -1.2567885269168115
    - type: nauc_ndcg_at_1_diff1
      value: 59.698357986628324
    - type: nauc_ndcg_at_1_max
      value: 36.19184535975336
    - type: nauc_ndcg_at_1_std
      value: -0.3732635881802827
    - type: nauc_ndcg_at_20_diff1
      value: 52.91626708083731
    - type: nauc_ndcg_at_20_max
      value: 37.3727463545816
    - type: nauc_ndcg_at_20_std
      value: 1.794148757644209
    - type: nauc_ndcg_at_3_diff1
      value: 55.41497362862388
    - type: nauc_ndcg_at_3_max
      value: 33.84606207970954
    - type: nauc_ndcg_at_3_std
      value: -1.5037390857368864
    - type: nauc_ndcg_at_5_diff1
      value: 55.561650253405716
    - type: nauc_ndcg_at_5_max
      value: 34.55478239305819
    - type: nauc_ndcg_at_5_std
      value: -2.6884049705546453
    - type: nauc_precision_at_1000_diff1
      value: -14.220538808948627
    - type: nauc_precision_at_1000_max
      value: 30.832501838042358
    - type: nauc_precision_at_1000_std
      value: 51.61025627560141
    - type: nauc_precision_at_100_diff1
      value: 4.4851329855278665
    - type: nauc_precision_at_100_max
      value: 42.019199750825834
    - type: nauc_precision_at_100_std
      value: 44.59826245592179
    - type: nauc_precision_at_10_diff1
      value: 31.128531158716914
    - type: nauc_precision_at_10_max
      value: 40.014303427714296
    - type: nauc_precision_at_10_std
      value: 18.66086480010028
    - type: nauc_precision_at_1_diff1
      value: 59.698357986628324
    - type: nauc_precision_at_1_max
      value: 36.19184535975336
    - type: nauc_precision_at_1_std
      value: -0.3732635881802827
    - type: nauc_precision_at_20_diff1
      value: 19.729038153986753
    - type: nauc_precision_at_20_max
      value: 43.018048891935095
    - type: nauc_precision_at_20_std
      value: 34.93305917294951
    - type: nauc_precision_at_3_diff1
      value: 47.69367211697757
    - type: nauc_precision_at_3_max
      value: 36.67930202405817
    - type: nauc_precision_at_3_std
      value: 10.015396528127898
    - type: nauc_precision_at_5_diff1
      value: 40.112747160286034
    - type: nauc_precision_at_5_max
      value: 36.522086663861955
    - type: nauc_precision_at_5_std
      value: 10.298695835086278
    - type: nauc_recall_at_1000_diff1
      value: 15.490555196437267
    - type: nauc_recall_at_1000_max
      value: 65.77784960137889
    - type: nauc_recall_at_1000_std
      value: 55.6956115779644
    - type: nauc_recall_at_100_diff1
      value: 31.420573665344186
    - type: nauc_recall_at_100_max
      value: 51.478517970815986
    - type: nauc_recall_at_100_std
      value: 21.8667299551994
    - type: nauc_recall_at_10_diff1
      value: 43.811043753262645
    - type: nauc_recall_at_10_max
      value: 35.251432327366395
    - type: nauc_recall_at_10_std
      value: -1.5842166373120425
    - type: nauc_recall_at_1_diff1
      value: 59.10174915043726
    - type: nauc_recall_at_1_max
      value: 33.416492729123185
    - type: nauc_recall_at_1_std
      value: -5.1952700488175045
    - type: nauc_recall_at_20_diff1
      value: 39.869129286563876
    - type: nauc_recall_at_20_max
      value: 43.8582881229973
    - type: nauc_recall_at_20_std
      value: 12.572449527758673
    - type: nauc_recall_at_3_diff1
      value: 51.9577529837867
    - type: nauc_recall_at_3_max
      value: 30.893069836408237
    - type: nauc_recall_at_3_std
      value: -1.959211453851564
    - type: nauc_recall_at_5_diff1
      value: 51.75561843815883
    - type: nauc_recall_at_5_max
      value: 32.15736743162173
    - type: nauc_recall_at_5_std
      value: -4.772474736542438
    - type: ndcg_at_1
      value: 42.667
    - type: ndcg_at_10
      value: 56.2
    - type: ndcg_at_100
      value: 60.260000000000005
    - type: ndcg_at_1000
      value: 61.483
    - type: ndcg_at_20
      value: 57.909
    - type: ndcg_at_3
      value: 51.711
    - type: ndcg_at_5
      value: 53.783
    - type: precision_at_1
      value: 42.667
    - type: precision_at_10
      value: 7.767
    - type: precision_at_100
      value: 0.993
    - type: precision_at_1000
      value: 0.11100000000000002
    - type: precision_at_20
      value: 4.3
    - type: precision_at_3
      value: 20.889
    - type: precision_at_5
      value: 13.866999999999999
    - type: recall_at_1
      value: 40.666999999999994
    - type: recall_at_10
      value: 70.244
    - type: recall_at_100
      value: 88.656
    - type: recall_at_1000
      value: 98.26700000000001
    - type: recall_at_20
      value: 76.589
    - type: recall_at_3
      value: 58.333
    - type: recall_at_5
      value: 63.24999999999999
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB SprintDuplicateQuestions (default)
      revision: d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46
      split: test
      type: mteb/sprintduplicatequestions-pairclassification
    metrics:
    - type: cosine_accuracy
      value: 99.73861386138614
    - type: cosine_accuracy_threshold
      value: 76.21053359407102
    - type: cosine_ap
      value: 92.55456011993165
    - type: cosine_f1
      value: 86.34953464322648
    - type: cosine_f1_threshold
      value: 76.21053359407102
    - type: cosine_precision
      value: 89.40042826552462
    - type: cosine_recall
      value: 83.5
    - type: dot_accuracy
      value: 99.73861386138614
    - type: dot_accuracy_threshold
      value: 76.21053204886738
    - type: dot_ap
      value: 92.55456011993165
    - type: dot_f1
      value: 86.34953464322648
    - type: dot_f1_threshold
      value: 76.21053204886738
    - type: dot_precision
      value: 89.40042826552462
    - type: dot_recall
      value: 83.5
    - type: euclidean_accuracy
      value: 99.73861386138614
    - type: euclidean_accuracy_threshold
      value: 68.9774713436305
    - type: euclidean_ap
      value: 92.55456011993165
    - type: euclidean_f1
      value: 86.34953464322648
    - type: euclidean_f1_threshold
      value: 68.9774713436305
    - type: euclidean_precision
      value: 89.40042826552462
    - type: euclidean_recall
      value: 83.5
    - type: main_score
      value: 92.55456011993165
    - type: manhattan_accuracy
      value: 99.72079207920792
    - type: manhattan_accuracy_threshold
      value: 1165.1618136773664
    - type: manhattan_ap
      value: 91.85005554989056
    - type: manhattan_f1
      value: 85.55327868852459
    - type: manhattan_f1_threshold
      value: 1169.146348835659
    - type: manhattan_precision
      value: 87.71008403361344
    - type: manhattan_recall
      value: 83.5
    - type: max_accuracy
      value: 99.73861386138614
    - type: max_ap
      value: 92.55456011993165
    - type: max_f1
      value: 86.34953464322648
    - type: max_precision
      value: 89.40042826552462
    - type: max_recall
      value: 83.5
    - type: similarity_accuracy
      value: 99.73861386138614
    - type: similarity_accuracy_threshold
      value: 76.21053359407102
    - type: similarity_ap
      value: 92.55456011993165
    - type: similarity_f1
      value: 86.34953464322648
    - type: similarity_f1_threshold
      value: 76.21053359407102
    - type: similarity_precision
      value: 89.40042826552462
    - type: similarity_recall
      value: 83.5
    task:
      type: PairClassification
  - dataset:
      config: default
      name: MTEB StackExchangeClustering (default)
      revision: 6cbc1f7b2bc0622f2e39d2c77fa502909748c259
      split: test
      type: mteb/stackexchange-clustering
    metrics:
    - type: main_score
      value: 51.975954494166075
    - type: v_measure
      value: 51.975954494166075
    - type: v_measure_std
      value: 4.557795328959378
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB StackExchangeClusteringP2P (default)
      revision: 815ca46b2622cec33ccafc3735d572c266efdb44
      split: test
      type: mteb/stackexchange-clustering-p2p
    metrics:
    - type: main_score
      value: 31.05220500143856
    - type: v_measure
      value: 31.05220500143856
    - type: v_measure_std
      value: 1.631365700671601
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB StackOverflowDupQuestions (default)
      revision: e185fbe320c72810689fc5848eb6114e1ef5ec69
      split: test
      type: mteb/stackoverflowdupquestions-reranking
    metrics:
    - type: main_score
      value: 43.96464532466761
    - type: map
      value: 43.96464532466761
    - type: mrr
      value: 44.37866055513114
    - type: nAUC_map_diff1
      value: 34.95374640225194
    - type: nAUC_map_max
      value: 14.380255206196887
    - type: nAUC_map_std
      value: 4.730399252834778
    - type: nAUC_mrr_diff1
      value: 34.07397654492457
    - type: nAUC_mrr_max
      value: 14.993446781842392
    - type: nAUC_mrr_std
      value: 4.8772543186698085
    task:
      type: Reranking
  - dataset:
      config: default
      name: MTEB SummEval (default)
      revision: cda12ad7615edc362dbf25a00fdd61d3b1eaf93c
      split: test
      type: mteb/summeval
    metrics:
    - type: cosine_pearson
      value: 30.188622278722267
    - type: cosine_spearman
      value: 29.781575832251296
    - type: dot_pearson
      value: 30.188623059165494
    - type: dot_spearman
      value: 29.790199244359933
    - type: main_score
      value: 29.781575832251296
    - type: pearson
      value: 30.188622278722267
    - type: spearman
      value: 29.781575832251296
    task:
      type: Summarization
  - dataset:
      config: default
      name: MTEB TRECCOVID (default)
      revision: bb9466bac8153a0349341eb1b22e06409e78ef4e
      split: test
      type: mteb/trec-covid
    metrics:
    - type: main_score
      value: 41.931000000000004
    - type: map_at_1
      value: 0.135
    - type: map_at_10
      value: 0.855
    - type: map_at_100
      value: 4.484
    - type: map_at_1000
      value: 11.498
    - type: map_at_20
      value: 1.478
    - type: map_at_3
      value: 0.336
    - type: map_at_5
      value: 0.503
    - type: mrr_at_1
      value: 56.00000000000001
    - type: mrr_at_10
      value: 64.68888888888888
    - type: mrr_at_100
      value: 65.45360311215018
    - type: mrr_at_1000
      value: 65.45360311215018
    - type: mrr_at_20
      value: 65.14263689526848
    - type: mrr_at_3
      value: 61.999999999999986
    - type: mrr_at_5
      value: 63.79999999999999
    - type: nauc_map_at_1000_diff1
      value: -4.815589003347539
    - type: nauc_map_at_1000_max
      value: 28.88855432647378
    - type: nauc_map_at_1000_std
      value: 39.02771004823413
    - type: nauc_map_at_100_diff1
      value: -6.763033694479543
    - type: nauc_map_at_100_max
      value: 18.19565344487221
    - type: nauc_map_at_100_std
      value: 20.443536772991617
    - type: nauc_map_at_10_diff1
      value: -8.372434926443631
    - type: nauc_map_at_10_max
      value: 21.02046545943698
    - type: nauc_map_at_10_std
      value: -7.766784888512321
    - type: nauc_map_at_1_diff1
      value: -11.726444894941686
    - type: nauc_map_at_1_max
      value: 28.098944623859012
    - type: nauc_map_at_1_std
      value: -8.992714094540933
    - type: nauc_map_at_20_diff1
      value: -9.035785185072163
    - type: nauc_map_at_20_max
      value: 18.506302521346917
    - type: nauc_map_at_20_std
      value: 0.8278714327228345
    - type: nauc_map_at_3_diff1
      value: -4.256022640946662
    - type: nauc_map_at_3_max
      value: 26.426928222979022
    - type: nauc_map_at_3_std
      value: -13.986031228577922
    - type: nauc_map_at_5_diff1
      value: -0.7971498197036772
    - type: nauc_map_at_5_max
      value: 28.264906005693895
    - type: nauc_map_at_5_std
      value: -9.864600825268974
    - type: nauc_mrr_at_1000_diff1
      value: -0.4613350723093772
    - type: nauc_mrr_at_1000_max
      value: 35.284828115023295
    - type: nauc_mrr_at_1000_std
      value: -1.7243940208394375
    - type: nauc_mrr_at_100_diff1
      value: -0.4613350723093772
    - type: nauc_mrr_at_100_max
      value: 35.284828115023295
    - type: nauc_mrr_at_100_std
      value: -1.7243940208394375
    - type: nauc_mrr_at_10_diff1
      value: -1.1774961298584272
    - type: nauc_mrr_at_10_max
      value: 34.502385968248525
    - type: nauc_mrr_at_10_std
      value: -1.8035899462487706
    - type: nauc_mrr_at_1_diff1
      value: 5.3401090165795555
    - type: nauc_mrr_at_1_max
      value: 38.54190324778555
    - type: nauc_mrr_at_1_std
      value: 6.129343629343582
    - type: nauc_mrr_at_20_diff1
      value: -0.16879233702767957
    - type: nauc_mrr_at_20_max
      value: 35.280815463947036
    - type: nauc_mrr_at_20_std
      value: -1.474513578251734
    - type: nauc_mrr_at_3_diff1
      value: -1.7688495464888414
    - type: nauc_mrr_at_3_max
      value: 32.94164095211376
    - type: nauc_mrr_at_3_std
      value: -6.87747405167847
    - type: nauc_mrr_at_5_diff1
      value: -1.9018790708586215
    - type: nauc_mrr_at_5_max
      value: 34.41650619460187
    - type: nauc_mrr_at_5_std
      value: -4.076153362419268
    - type: nauc_ndcg_at_1000_diff1
      value: -7.11758000334035
    - type: nauc_ndcg_at_1000_max
      value: 26.33768168184003
    - type: nauc_ndcg_at_1000_std
      value: 31.441892174911988
    - type: nauc_ndcg_at_100_diff1
      value: 5.945901478997322
    - type: nauc_ndcg_at_100_max
      value: 25.317381446915604
    - type: nauc_ndcg_at_100_std
      value: 25.557558325471348
    - type: nauc_ndcg_at_10_diff1
      value: 4.325756905707739
    - type: nauc_ndcg_at_10_max
      value: 25.232156948625345
    - type: nauc_ndcg_at_10_std
      value: 3.8168250010393354
    - type: nauc_ndcg_at_1_diff1
      value: 7.471852610030701
    - type: nauc_ndcg_at_1_max
      value: 32.29068577277381
    - type: nauc_ndcg_at_1_std
      value: 2.819515523712047
    - type: nauc_ndcg_at_20_diff1
      value: 4.7861088304576205
    - type: nauc_ndcg_at_20_max
      value: 23.59279585898106
    - type: nauc_ndcg_at_20_std
      value: 11.465220742781467
    - type: nauc_ndcg_at_3_diff1
      value: 7.68150652123663
    - type: nauc_ndcg_at_3_max
      value: 28.90636434919282
    - type: nauc_ndcg_at_3_std
      value: -6.057560500143322
    - type: nauc_ndcg_at_5_diff1
      value: 7.174553606416769
    - type: nauc_ndcg_at_5_max
      value: 28.32458435468132
    - type: nauc_ndcg_at_5_std
      value: -1.3744649180278934
    - type: nauc_precision_at_1000_diff1
      value: 5.584231801519359
    - type: nauc_precision_at_1000_max
      value: 28.002541879211844
    - type: nauc_precision_at_1000_std
      value: 33.27730700949564
    - type: nauc_precision_at_100_diff1
      value: 6.346489878855957
    - type: nauc_precision_at_100_max
      value: 26.95277497012729
    - type: nauc_precision_at_100_std
      value: 28.645191802637154
    - type: nauc_precision_at_10_diff1
      value: 2.3502489108128826
    - type: nauc_precision_at_10_max
      value: 27.281466664077225
    - type: nauc_precision_at_10_std
      value: 6.9364872825671675
    - type: nauc_precision_at_1_diff1
      value: 5.3401090165795555
    - type: nauc_precision_at_1_max
      value: 38.54190324778555
    - type: nauc_precision_at_1_std
      value: 6.129343629343582
    - type: nauc_precision_at_20_diff1
      value: 3.839972337238674
    - type: nauc_precision_at_20_max
      value: 27.22170962215984
    - type: nauc_precision_at_20_std
      value: 17.733276643490875
    - type: nauc_precision_at_3_diff1
      value: 6.598478731146103
    - type: nauc_precision_at_3_max
      value: 27.451625444866444
    - type: nauc_precision_at_3_std
      value: -9.193408499566358
    - type: nauc_precision_at_5_diff1
      value: 6.452908005912722
    - type: nauc_precision_at_5_max
      value: 31.655017610453136
    - type: nauc_precision_at_5_std
      value: 0.8674440814886519
    - type: nauc_recall_at_1000_diff1
      value: -9.719516261417361
    - type: nauc_recall_at_1000_max
      value: 25.06055928220044
    - type: nauc_recall_at_1000_std
      value: 33.23548730908827
    - type: nauc_recall_at_100_diff1
      value: -11.518579756795685
    - type: nauc_recall_at_100_max
      value: 15.646642192110278
    - type: nauc_recall_at_100_std
      value: 19.192418060016227
    - type: nauc_recall_at_10_diff1
      value: -13.48419022233322
    - type: nauc_recall_at_10_max
      value: 19.013312050968338
    - type: nauc_recall_at_10_std
      value: -7.2511820688363855
    - type: nauc_recall_at_1_diff1
      value: -11.726444894941686
    - type: nauc_recall_at_1_max
      value: 28.098944623859012
    - type: nauc_recall_at_1_std
      value: -8.992714094540933
    - type: nauc_recall_at_20_diff1
      value: -14.398393925439896
    - type: nauc_recall_at_20_max
      value: 17.2065810175314
    - type: nauc_recall_at_20_std
      value: 1.3585608911222675
    - type: nauc_recall_at_3_diff1
      value: -8.290744359014761
    - type: nauc_recall_at_3_max
      value: 21.881683099047937
    - type: nauc_recall_at_3_std
      value: -18.361115623762856
    - type: nauc_recall_at_5_diff1
      value: -5.286572307527278
    - type: nauc_recall_at_5_max
      value: 25.256200846766614
    - type: nauc_recall_at_5_std
      value: -10.683496663885045
    - type: ndcg_at_1
      value: 51.0
    - type: ndcg_at_10
      value: 41.931000000000004
    - type: ndcg_at_100
      value: 31.928
    - type: ndcg_at_1000
      value: 29.409000000000002
    - type: ndcg_at_20
      value: 40.937
    - type: ndcg_at_3
      value: 47.704
    - type: ndcg_at_5
      value: 44.912
    - type: precision_at_1
      value: 56.00000000000001
    - type: precision_at_10
      value: 44.4
    - type: precision_at_100
      value: 33.14
    - type: precision_at_1000
      value: 14.198
    - type: precision_at_20
      value: 43.6
    - type: precision_at_3
      value: 50.0
    - type: precision_at_5
      value: 46.800000000000004
    - type: recall_at_1
      value: 0.135
    - type: recall_at_10
      value: 1.061
    - type: recall_at_100
      value: 7.166
    - type: recall_at_1000
      value: 28.378999999999998
    - type: recall_at_20
      value: 1.9980000000000002
    - type: recall_at_3
      value: 0.367
    - type: recall_at_5
      value: 0.5780000000000001
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB Touche2020 (default)
      revision: a34f9a33db75fa0cbb21bb5cfc3dae8dc8bec93f
      split: test
      type: mteb/touche2020
    metrics:
    - type: main_score
      value: 19.628999999999998
    - type: map_at_1
      value: 1.7770000000000001
    - type: map_at_10
      value: 7.313
    - type: map_at_100
      value: 11.988
    - type: map_at_1000
      value: 13.514999999999999
    - type: map_at_20
      value: 8.796
    - type: map_at_3
      value: 3.2169999999999996
    - type: map_at_5
      value: 4.601
    - type: mrr_at_1
      value: 20.408163265306122
    - type: mrr_at_10
      value: 37.27891156462584
    - type: mrr_at_100
      value: 38.6072477967195
    - type: mrr_at_1000
      value: 38.6072477967195
    - type: mrr_at_20
      value: 38.08898163608468
    - type: mrr_at_3
      value: 32.6530612244898
    - type: mrr_at_5
      value: 36.734693877551024
    - type: nauc_map_at_1000_diff1
      value: 10.570624698916106
    - type: nauc_map_at_1000_max
      value: -31.294202007427387
    - type: nauc_map_at_1000_std
      value: -14.937652619312106
    - type: nauc_map_at_100_diff1
      value: 11.868722412438842
    - type: nauc_map_at_100_max
      value: -30.865420770703604
    - type: nauc_map_at_100_std
      value: -20.422724780426734
    - type: nauc_map_at_10_diff1
      value: 0.041493306597119875
    - type: nauc_map_at_10_max
      value: -40.396176612553965
    - type: nauc_map_at_10_std
      value: -28.509321814611592
    - type: nauc_map_at_1_diff1
      value: 7.127954407520293
    - type: nauc_map_at_1_max
      value: -35.3892053151105
    - type: nauc_map_at_1_std
      value: -19.49733359371979
    - type: nauc_map_at_20_diff1
      value: 6.085228939912852
    - type: nauc_map_at_20_max
      value: -32.67429246305332
    - type: nauc_map_at_20_std
      value: -25.673480187765403
    - type: nauc_map_at_3_diff1
      value: 4.0380003376283735
    - type: nauc_map_at_3_max
      value: -38.314578563212734
    - type: nauc_map_at_3_std
      value: -22.944579429221008
    - type: nauc_map_at_5_diff1
      value: 1.279980605614138
    - type: nauc_map_at_5_max
      value: -43.13533497355372
    - type: nauc_map_at_5_std
      value: -27.960528588143113
    - type: nauc_mrr_at_1000_diff1
      value: -1.7632046089447704
    - type: nauc_mrr_at_1000_max
      value: -37.614642648352415
    - type: nauc_mrr_at_1000_std
      value: -20.453733496900504
    - type: nauc_mrr_at_100_diff1
      value: -1.7632046089447704
    - type: nauc_mrr_at_100_max
      value: -37.614642648352415
    - type: nauc_mrr_at_100_std
      value: -20.453733496900504
    - type: nauc_mrr_at_10_diff1
      value: -0.8838352304131204
    - type: nauc_mrr_at_10_max
      value: -36.76914151415433
    - type: nauc_mrr_at_10_std
      value: -20.287505509157537
    - type: nauc_mrr_at_1_diff1
      value: -1.0597262941161223
    - type: nauc_mrr_at_1_max
      value: -30.53828106000604
    - type: nauc_mrr_at_1_std
      value: -22.137640176831137
    - type: nauc_mrr_at_20_diff1
      value: -2.1539344598000856
    - type: nauc_mrr_at_20_max
      value: -37.69131983804487
    - type: nauc_mrr_at_20_std
      value: -20.36070687538437
    - type: nauc_mrr_at_3_diff1
      value: -1.893438268797217
    - type: nauc_mrr_at_3_max
      value: -36.28440192297394
    - type: nauc_mrr_at_3_std
      value: -19.991286112256187
    - type: nauc_mrr_at_5_diff1
      value: -2.0922206601916677
    - type: nauc_mrr_at_5_max
      value: -39.41732522875257
    - type: nauc_mrr_at_5_std
      value: -21.491524754378055
    - type: nauc_ndcg_at_1000_diff1
      value: 14.479754053162955
    - type: nauc_ndcg_at_1000_max
      value: -32.10581012377326
    - type: nauc_ndcg_at_1000_std
      value: 13.35087951605272
    - type: nauc_ndcg_at_100_diff1
      value: 16.80964326984191
    - type: nauc_ndcg_at_100_max
      value: -32.33147471694196
    - type: nauc_ndcg_at_100_std
      value: -5.868369097951216
    - type: nauc_ndcg_at_10_diff1
      value: 2.289700885976584
    - type: nauc_ndcg_at_10_max
      value: -38.216881297234615
    - type: nauc_ndcg_at_10_std
      value: -20.20137297942159
    - type: nauc_ndcg_at_1_diff1
      value: 1.7910846362434194
    - type: nauc_ndcg_at_1_max
      value: -29.634269914775903
    - type: nauc_ndcg_at_1_std
      value: -21.190379097876075
    - type: nauc_ndcg_at_20_diff1
      value: 8.806996779903233
    - type: nauc_ndcg_at_20_max
      value: -33.96095191892843
    - type: nauc_ndcg_at_20_std
      value: -20.125770326355852
    - type: nauc_ndcg_at_3_diff1
      value: -1.091472840622981
    - type: nauc_ndcg_at_3_max
      value: -33.11081822038949
    - type: nauc_ndcg_at_3_std
      value: -16.877176763631756
    - type: nauc_ndcg_at_5_diff1
      value: -1.082579219115309
    - type: nauc_ndcg_at_5_max
      value: -41.3743136016136
    - type: nauc_ndcg_at_5_std
      value: -20.715523834992034
    - type: nauc_precision_at_1000_diff1
      value: -6.323556781007291
    - type: nauc_precision_at_1000_max
      value: 34.58199835082229
    - type: nauc_precision_at_1000_std
      value: 60.77523423450688
    - type: nauc_precision_at_100_diff1
      value: 14.226845553421732
    - type: nauc_precision_at_100_max
      value: -1.7650024651415825
    - type: nauc_precision_at_100_std
      value: 26.54538052469
    - type: nauc_precision_at_10_diff1
      value: 5.217036148150041
    - type: nauc_precision_at_10_max
      value: -26.095792438253195
    - type: nauc_precision_at_10_std
      value: -16.80604019839076
    - type: nauc_precision_at_1_diff1
      value: -1.0597262941161223
    - type: nauc_precision_at_1_max
      value: -30.53828106000604
    - type: nauc_precision_at_1_std
      value: -22.137640176831137
    - type: nauc_precision_at_20_diff1
      value: 11.168974829140431
    - type: nauc_precision_at_20_max
      value: -16.16782573302428
    - type: nauc_precision_at_20_std
      value: -11.053623767620662
    - type: nauc_precision_at_3_diff1
      value: 0.947307929112776
    - type: nauc_precision_at_3_max
      value: -31.05930383886252
    - type: nauc_precision_at_3_std
      value: -17.102851167484616
    - type: nauc_precision_at_5_diff1
      value: 0.9266007195538144
    - type: nauc_precision_at_5_max
      value: -37.581387889994
    - type: nauc_precision_at_5_std
      value: -19.690150307959293
    - type: nauc_recall_at_1000_diff1
      value: 16.984317473823676
    - type: nauc_recall_at_1000_max
      value: -30.587646676977936
    - type: nauc_recall_at_1000_std
      value: 57.48516026941197
    - type: nauc_recall_at_100_diff1
      value: 21.886301436834625
    - type: nauc_recall_at_100_max
      value: -28.07071340770586
    - type: nauc_recall_at_100_std
      value: -0.11337935267596243
    - type: nauc_recall_at_10_diff1
      value: 1.6250953111186457
    - type: nauc_recall_at_10_max
      value: -38.0654848945014
    - type: nauc_recall_at_10_std
      value: -28.014162803875355
    - type: nauc_recall_at_1_diff1
      value: 7.127954407520293
    - type: nauc_recall_at_1_max
      value: -35.3892053151105
    - type: nauc_recall_at_1_std
      value: -19.49733359371979
    - type: nauc_recall_at_20_diff1
      value: 10.635038702861864
    - type: nauc_recall_at_20_max
      value: -30.098601211040904
    - type: nauc_recall_at_20_std
      value: -22.936538824524817
    - type: nauc_recall_at_3_diff1
      value: 3.5564807135760255
    - type: nauc_recall_at_3_max
      value: -38.0587512734278
    - type: nauc_recall_at_3_std
      value: -21.84694412548793
    - type: nauc_recall_at_5_diff1
      value: -0.7556448626996317
    - type: nauc_recall_at_5_max
      value: -46.540574319477905
    - type: nauc_recall_at_5_std
      value: -30.56076072609364
    - type: ndcg_at_1
      value: 16.326999999999998
    - type: ndcg_at_10
      value: 19.628999999999998
    - type: ndcg_at_100
      value: 30.853
    - type: ndcg_at_1000
      value: 42.881
    - type: ndcg_at_20
      value: 20.232
    - type: ndcg_at_3
      value: 18.093999999999998
    - type: ndcg_at_5
      value: 19.089
    - type: precision_at_1
      value: 20.408
    - type: precision_at_10
      value: 20.0
    - type: precision_at_100
      value: 7.204000000000001
    - type: precision_at_1000
      value: 1.488
    - type: precision_at_20
      value: 14.796000000000001
    - type: precision_at_3
      value: 20.408
    - type: precision_at_5
      value: 21.224
    - type: recall_at_1
      value: 1.7770000000000001
    - type: recall_at_10
      value: 14.056
    - type: recall_at_100
      value: 43.388
    - type: recall_at_1000
      value: 80.384
    - type: recall_at_20
      value: 19.73
    - type: recall_at_3
      value: 4.444
    - type: recall_at_5
      value: 7.742
    task:
      type: Retrieval
  - dataset:
      config: default
      name: MTEB ToxicConversationsClassification (default)
      revision: edfaf9da55d3dd50d43143d90c1ac476895ae6de
      split: test
      type: mteb/toxic_conversations_50k
    metrics:
    - type: accuracy
      value: 70.4345703125
    - type: ap
      value: 13.363055996397314
    - type: ap_weighted
      value: 13.363055996397314
    - type: f1
      value: 53.71432014147602
    - type: f1_weighted
      value: 76.97001715054664
    - type: main_score
      value: 70.4345703125
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB TweetSentimentExtractionClassification (default)
      revision: d604517c81ca91fe16a244d1248fc021f9ecee7a
      split: test
      type: mteb/tweet_sentiment_extraction
    metrics:
    - type: accuracy
      value: 56.58460667798529
    - type: f1
      value: 56.81197245206573
    - type: f1_weighted
      value: 56.13824904215221
    - type: main_score
      value: 56.58460667798529
    task:
      type: Classification
  - dataset:
      config: default
      name: MTEB TwentyNewsgroupsClustering (default)
      revision: 6125ec4e24fa026cec8a478383ee943acfbd5449
      split: test
      type: mteb/twentynewsgroups-clustering
    metrics:
    - type: main_score
      value: 36.773580338060434
    - type: v_measure
      value: 36.773580338060434
    - type: v_measure_std
      value: 2.1187678513989585
    task:
      type: Clustering
  - dataset:
      config: default
      name: MTEB TwitterSemEval2015 (default)
      revision: 70970daeab8776df92f5ea462b6173c0b46fd2d1
      split: test
      type: mteb/twittersemeval2015-pairclassification
    metrics:
    - type: cosine_accuracy
      value: 81.8024676640639
    - type: cosine_accuracy_threshold
      value: 72.72156987862404
    - type: cosine_ap
      value: 58.626480762860155
    - type: cosine_f1
      value: 56.335306196400516
    - type: cosine_f1_threshold
      value: 63.90516602618299
    - type: cosine_precision
      value: 51.94920917799064
    - type: cosine_recall
      value: 61.53034300791557
    - type: dot_accuracy
      value: 81.8024676640639
    - type: dot_accuracy_threshold
      value: 72.72157129741353
    - type: dot_ap
      value: 58.626480927811144
    - type: dot_f1
      value: 56.335306196400516
    - type: dot_f1_threshold
      value: 63.90516471984052
    - type: dot_precision
      value: 51.94920917799064
    - type: dot_recall
      value: 61.53034300791557
    - type: euclidean_accuracy
      value: 81.8024676640639
    - type: euclidean_accuracy_threshold
      value: 73.86261650981828
    - type: euclidean_ap
      value: 58.626478566508865
    - type: euclidean_f1
      value: 56.335306196400516
    - type: euclidean_f1_threshold
      value: 84.96450225024545
    - type: euclidean_precision
      value: 51.94920917799064
    - type: euclidean_recall
      value: 61.53034300791557
    - type: main_score
      value: 59.15380842181217
    - type: manhattan_accuracy
      value: 81.88591524110389
    - type: manhattan_accuracy_threshold
      value: 1181.9765670520376
    - type: manhattan_ap
      value: 59.15380842181217
    - type: manhattan_f1
      value: 56.939975590813276
    - type: manhattan_f1_threshold
      value: 1437.7120667882991
    - type: manhattan_precision
      value: 49.12885314953092
    - type: manhattan_recall
      value: 67.70448548812665
    - type: max_accuracy
      value: 81.88591524110389
    - type: max_ap
      value: 59.15380842181217
    - type: max_f1
      value: 56.939975590813276
    - type: max_precision
      value: 51.94920917799064
    - type: max_recall
      value: 67.70448548812665
    - type: similarity_accuracy
      value: 81.8024676640639
    - type: similarity_accuracy_threshold
      value: 72.72156987862404
    - type: similarity_ap
      value: 58.626480762860155
    - type: similarity_f1
      value: 56.335306196400516
    - type: similarity_f1_threshold
      value: 63.90516602618299
    - type: similarity_precision
      value: 51.94920917799064
    - type: similarity_recall
      value: 61.53034300791557
    task:
      type: PairClassification
  - dataset:
      config: default
      name: MTEB TwitterURLCorpus (default)
      revision: 8b6510b0b1fa4e4c4f879467980e9be563ec1cdf
      split: test
      type: mteb/twitterurlcorpus-pairclassification
    metrics:
    - type: cosine_accuracy
      value: 87.36174176271976
    - type: cosine_accuracy_threshold
      value: 60.648304783781334
    - type: cosine_ap
      value: 82.50029451974736
    - type: cosine_f1
      value: 74.85008308648219
    - type: cosine_f1_threshold
      value: 54.0321362274791
    - type: cosine_precision
      value: 70.50496801415544
    - type: cosine_recall
      value: 79.76593778872805
    - type: dot_accuracy
      value: 87.36174176271976
    - type: dot_accuracy_threshold
      value: 60.64830361623812
    - type: dot_ap
      value: 82.50033093545332
    - type: dot_f1
      value: 74.85008308648219
    - type: dot_f1_threshold
      value: 54.03213667524275
    - type: dot_precision
      value: 70.50496801415544
    - type: dot_recall
      value: 79.76593778872805
    - type: euclidean_accuracy
      value: 87.36174176271976
    - type: euclidean_accuracy_threshold
      value: 88.714930435829
    - type: euclidean_ap
      value: 82.50029575944534
    - type: euclidean_f1
      value: 74.85008308648219
    - type: euclidean_f1_threshold
      value: 95.8831206738777
    - type: euclidean_precision
      value: 70.50496801415544
    - type: euclidean_recall
      value: 79.76593778872805
    - type: main_score
      value: 82.79743429849837
    - type: manhattan_accuracy
      value: 87.51309814879497
    - type: manhattan_accuracy_threshold
      value: 1458.0611945921191
    - type: manhattan_ap
      value: 82.79743429849837
    - type: manhattan_f1
      value: 75.14295079578979
    - type: manhattan_f1_threshold
      value: 1571.1923710026895
    - type: manhattan_precision
      value: 71.29725620291659
    - type: manhattan_recall
      value: 79.4271635355713
    - type: max_accuracy
      value: 87.51309814879497
    - type: max_ap
      value: 82.79743429849837
    - type: max_f1
      value: 75.14295079578979
    - type: max_precision
      value: 71.29725620291659
    - type: max_recall
      value: 79.76593778872805
    - type: similarity_accuracy
      value: 87.36174176271976
    - type: similarity_accuracy_threshold
      value: 60.648304783781334
    - type: similarity_ap
      value: 82.50029451974736
    - type: similarity_f1
      value: 74.85008308648219
    - type: similarity_f1_threshold
      value: 54.0321362274791
    - type: similarity_precision
      value: 70.50496801415544
    - type: similarity_recall
      value: 79.76593778872805
    task:
      type: PairClassification
model_name: potion-base-32M
tags:
- embeddings
- static-embeddings
- mteb
- sentence-transformers
---


# potion-base-32M Model Card

<div align="center">
  <img width="35%" alt="Model2Vec logo" src="https://raw.githubusercontent.com/MinishLab/model2vec/main/assets/images/logo_v2.png">
</div>


This [Model2Vec](https://github.com/MinishLab/model2vec) model is pre-trained using [Tokenlearn](https://github.com/MinishLab/tokenlearn). It is a distilled version of the [baai/bge-base-en-v1.5](https://huggingface.co/baai/bge-base-en-v1.5) Sentence Transformer. It uses static embeddings, allowing text embeddings to be computed orders of magnitude faster on both GPU and CPU. It is designed for applications where computational resources are limited or where real-time performance is critical. It uses a larger vocabulary size than the [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) model which can be beneficial for tasks that require a larger vocabulary.



## Installation

Install model2vec using pip:
```
pip install model2vec
```

## Usage
Load this model using the `from_pretrained` method:
```python
from model2vec import StaticModel

# Load a pretrained Model2Vec model
model = StaticModel.from_pretrained("minishlab/potion-base-32M")

# Compute text embeddings
embeddings = model.encode(["Example sentence"])
```


## How it works

Model2vec creates a small, static model that outperforms other static embedding models by a large margin on all tasks on [MTEB](https://huggingface.co/spaces/mteb/leaderboard). This model is pre-trained using [Tokenlearn](https://github.com/MinishLab/tokenlearn). It's created using the following steps:
- Distillation: first, a model is distilled from a sentence transformer model using Model2Vec.
- Training data creation: the sentence transformer model is used to create training data by creating mean output embeddings on a large corpus.
- Training: the distilled model is trained on the training data using Tokenlearn.
- Post-training re-regularization: after training, the model is re-regularized by weighting the tokens based on their frequency, applying PCA, and finally applying [SIF weighting](https://openreview.net/pdf?id=SyK00v5xx).



## Results

The results for this model are shown in the table below. The full Model2Vec results for all models can be found on the [Model2Vec results page](https://github.com/MinishLab/model2vec/blob/main/results/README.md).
```
Average (All)                               52.46
Average (MTEB)                              51.66
Classification                              65.97
Clustering                                  35.29
PairClassification                          78.17
Reranking                                   50.92
Retrieval                                   33.52
STS                                         74.22
Summarization                               29.78
PEARL                                       55.37
WordSim                                     55.15
```


## Additional Resources

- [All Model2Vec models on the hub](https://huggingface.co/models?library=model2vec)
- [Model2Vec Repo](https://github.com/MinishLab/model2vec)
- [Tokenlearn repo](https://github.com/MinishLab/tokenlearn)
- [Model2Vec Results](https://github.com/MinishLab/model2vec/blob/main/results/README.md)
- [Model2Vec Tutorials](https://github.com/MinishLab/model2vec/tree/main/tutorials)

## Library Authors

Model2Vec was developed by the [Minish Lab](https://github.com/MinishLab) team consisting of [Stephan Tulkens](https://github.com/stephantul) and [Thomas van Dongen](https://github.com/Pringled).

## Citation

Please cite the [Model2Vec repository](https://github.com/MinishLab/model2vec) if you use this model in your work.
```
@software{minishlab2024model2vec,
  authors = {Stephan Tulkens and Thomas van Dongen},
  title = {Model2Vec: The Fastest State-of-the-Art Static Embeddings in the World},
  year = {2024},
  url = {https://github.com/MinishLab/model2vec}
}
```