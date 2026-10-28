# M2 3D와 원판 기하 결과

기본값: **tracked N=10, bench_rep_guard=False**. 3D는 진단 출력이며 판정에는 미사용.
원판 69개 기록·결과와 횟수 61개는 M0·M1·M1b 모든 전체 설정과 동일(실행 시간 필드 제외). 원판 rows는 gzip 해제 후 바이트 일치. 3D 확인 이미지 69장을 모두 검토했다.

키 추정 불가 14개, 150~200cm 밖 29개다. 3D 이미지 이상 관찰은 31개이며, 특히 벤치 복원은 판정용 정확도로 검증되지 않았다.

## 벤치 16개: guard 전후

|설정|coverage|리프터 오류 안|리프터 오류 밖|오류 합집합|
|---|---:|---:|---:|---:|
|M1b n10 guard on|0.8872|0|1|1|
|M2 n10 guard off|0.9854|0|1|1|

|영상|coverage 전|coverage 후|안 오류 프레임|밖 오류 프레임|확인 이미지|
|---|---:|---:|---|---|---|
|benchpress/bench press_31.mp4|1.0000|1.0000|[]|[]|bench_n10/images/benchpress__bench press_31.mp4.jpg|
|benchpress/bench press_40.mp4|0.9401|1.0000|[]|[]|bench_n10/images/benchpress__bench press_40.mp4.jpg|
|benchpress/stu10_11.mp4|N/A|N/A|[]|[0]|bench_n10/images/benchpress__stu10_11.mp4.jpg|
|benchpress/stu8_15.mp4|0.9953|1.0000|[]|[]|bench_n10/images/benchpress__stu8_15.mp4.jpg|
|benchpress/stu8_7.mp4|1.0000|1.0000|[]|[]|bench_n10/images/benchpress__stu8_7.mp4.jpg|
|benchpress/test153.mp4|N/A|N/A|[]|[]|bench_n10/images/benchpress__test153.mp4.jpg|
|benchpress/bench press_61.mp4|1.0000|1.0000|[]|[]|bench_n10/images/benchpress__bench press_61.mp4.jpg|
|benchpress/stu1_6_4.mp4|0.0000|1.0000|[]|[]|bench_n10/images/benchpress__stu1_6_4.mp4.jpg|
|benchpress/stu7_12_2.mp4|0.5769|1.0000|[]|[]|bench_n10/images/benchpress__stu7_12_2.mp4.jpg|
|benchpress/stu7_7.mp4|0.9732|0.9911|[]|[]|bench_n10/images/benchpress__stu7_7.mp4.jpg|
|benchpress/stu8_6_2.mp4|0.8316|0.8418|[]|[]|bench_n10/images/benchpress__stu8_6_2.mp4.jpg|
|benchpress/bench_01.mp4|0.9767|1.0000|[]|[]|bench_n10/images/benchpress__bench_01.mp4.jpg|
|benchpress/bench_02.mp4|1.0000|1.0000|[]|[]|bench_n10/images/benchpress__bench_02.mp4.jpg|
|benchpress/bench_03.mp4|1.0000|1.0000|[]|[]|bench_n10/images/benchpress__bench_03.mp4.jpg|
|team9/benchpress/stu9_10.mp4|0.8986|0.9797|[]|[]|bench_n10/images/team9__benchpress__stu9_10.mp4.jpg|
|team9/benchpress/stu2_10_edited.mp4|1.0000|1.0000|[]|[]|bench_n10/images/team9__benchpress__stu2_10_edited.mp4.jpg|

## 과거 lifter_review의 완료 rep 안/밖 분류

|설정|안 오류 영상|밖 오류 영상|합집합|
|---|---:|---:|---:|
|M0|2|4|6|
|M1 lightweight|2|3|5|
|M1 balanced|1|1|2|
|M1b n3|1|2|3|
|M1b n5|1|2|3|
|M1b n10|1|2|3|
|M1b tiny_m|3|3|6|
|M1b bench_guard|0|1|1|
|M2 bench guard off|0|1|1|
|M2 all selected2d|1|2|3|

### 오류 표본과 이미지

|설정|영상|안 프레임|밖 프레임|이미지|
|---|---|---|---|---|
|M0|benchpress/stu10_11.mp4|[]|[359, 719, 1078, 1438]|M0/images/benchpress__stu10_11.mp4.jpg|
|M0|benchpress/test153.mp4|[]|[0, 239, 299]|M0/images/benchpress__test153.mp4.jpg|
|M0|squat/stu6_11_2.mp4|[]|[127, 159]|M0/images/squat__stu6_11_2.mp4.jpg|
|M0|squat/stu6_11_3.mp4|[]|[0, 38]|M0/images/squat__stu6_11_3.mp4.jpg|
|M0|deadlift/GB_S04_R03.mp4|[36]|[]|M0/images/deadlift__GB_S04_R03.mp4.jpg|
|M0|team9/benchpress/stu9_10.mp4|[243, 401, 558, 740, 904, 1085]|[]|M0/images/team9__benchpress__stu9_10.mp4.jpg|
|M1 lightweight|squat/val1235.mp4|[52, 232]|[]|M1_lightweight/images/squat__val1235.mp4.jpg|
|M1 lightweight|benchpress/stu10_11.mp4|[]|[0]|M1_lightweight/images/benchpress__stu10_11.mp4.jpg|
|M1 lightweight|benchpress/test153.mp4|[]|[60]|M1_lightweight/images/benchpress__test153.mp4.jpg|
|M1 lightweight|squat/stu6_11_2.mp4|[]|[95]|M1_lightweight/images/squat__stu6_11_2.mp4.jpg|
|M1 lightweight|team9/benchpress/stu9_10.mp4|[401, 558, 740]|[]|M1_lightweight/images/team9__benchpress__stu9_10.mp4.jpg|
|M1 balanced|squat/val1235.mp4|[0, 105, 179]|[]|M1_balanced/images/squat__val1235.mp4.jpg|
|M1 balanced|benchpress/stu10_11.mp4|[]|[0]|M1_balanced/images/benchpress__stu10_11.mp4.jpg|
|M1b n3|squat/val1235.mp4|[0]|[]|M1b_n3/images/squat__val1235.mp4.jpg|
|M1b n3|benchpress/stu10_11.mp4|[]|[0]|M1b_n3/images/benchpress__stu10_11.mp4.jpg|
|M1b n3|squat/stu6_11_2.mp4|[]|[95]|M1b_n3/images/squat__stu6_11_2.mp4.jpg|
|M1b n5|squat/val1235.mp4|[0]|[]|M1b_n5/images/squat__val1235.mp4.jpg|
|M1b n5|benchpress/stu10_11.mp4|[]|[0, 1438]|M1b_n5/images/benchpress__stu10_11.mp4.jpg|
|M1b n5|squat/stu6_11_2.mp4|[]|[95]|M1b_n5/images/squat__stu6_11_2.mp4.jpg|
|M1b n10|squat/val1235.mp4|[0]|[]|M1b_n10/images/squat__val1235.mp4.jpg|
|M1b n10|benchpress/stu10_11.mp4|[]|[0]|M1b_n10/images/benchpress__stu10_11.mp4.jpg|
|M1b n10|squat/stu6_11_2.mp4|[]|[95]|M1b_n10/images/squat__stu6_11_2.mp4.jpg|
|M1b tiny_m|squat/val1235.mp4|[52, 105, 179]|[]|M1b_tiny_m/images/squat__val1235.mp4.jpg|
|M1b tiny_m|benchpress/stu10_11.mp4|[]|[0]|M1b_tiny_m/images/benchpress__stu10_11.mp4.jpg|
|M1b tiny_m|benchpress/test153.mp4|[]|[60]|M1b_tiny_m/images/benchpress__test153.mp4.jpg|
|M1b tiny_m|squat/stu6_11_2.mp4|[]|[95]|M1b_tiny_m/images/squat__stu6_11_2.mp4.jpg|
|M1b tiny_m|deadlift/LLK_S01_R03.mp4|[59]|[]|M1b_tiny_m/images/deadlift__LLK_S01_R03.mp4.jpg|
|M1b tiny_m|team9/benchpress/stu9_10.mp4|[558]|[]|M1b_tiny_m/images/team9__benchpress__stu9_10.mp4.jpg|
|M1b bench_guard|benchpress/stu10_11.mp4|[]|[0]|M1b_bench_guard/images/benchpress__stu10_11.mp4.jpg|
|M2 bench guard off|benchpress/stu10_11.mp4|[]|[0]|bench_n10/images/benchpress__stu10_11.mp4.jpg|
|M2 all selected2d|squat/val1235.mp4|[0]|[]|M1b_n10/images/squat__val1235.mp4.jpg|
|M2 all selected2d|benchpress/stu10_11.mp4|[]|[0]|bench_n10/images/benchpress__stu10_11.mp4.jpg|
|M2 all selected2d|squat/stu6_11_2.mp4|[]|[95]|M1b_n10/images/squat__stu6_11_2.mp4.jpg|

## 3D 요약

|종목|영상|뼈 CV 전|뼈 CV 후|가속도 전 m/s²|가속도 후 m/s²|3D 처리 초|
|---|---:|---:|---:|---:|---:|---:|
|all|69|0.0583|0.0000|4.3940|1.1147|16.1079|
|squat|13|0.0524|0.0000|5.1827|0.9695|3.9521|
|benchpress|16|0.1238|0.0000|11.6958|1.5260|6.5066|
|deadlift|40|0.0506|0.0000|3.4984|1.0353|5.6492|

CV는 세트별 모든 tree 뼈의 중앙값, 가속도는 영상별 중앙값들의 중앙값이다. 세트별 목표 길이가 달라도 고정 후 세트 내 CV는 수치 오차 수준이다. 상세 분모와 뼈별 값은 영상별 JSON에 있다.

## 영상별 기하·분절

|영상|각도 °|거리 m|키 cm|넙다리 cm|정강이 cm|몸통 cm|위팔 cm|아래팔 cm|서 있는 프레임|키 보정|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|squat/squat_11.mp4|4.7030|1.7111|157.4557|44.3286|43.8279|44.6658|29.6544|20.1612|55|True|
|squat/stu4_69.mp4|11.9498|2.0467|174.9211|50.4879|47.4041|51.9643|30.7616|23.7491|140|True|
|squat/val1235.mp4|0.0000|1.5713|143.6630|39.5649|38.5277|42.7842|21.8032|14.8338|30|True|
|squat/val1256.mp4|0.0000|1.9059|187.4859|54.4965|57.4356|47.4314|23.9008|14.9159|81|True|
|benchpress/bench press_31.mp4|60.6537|1.5902|N/A|47.0517|37.5897|40.6522|22.7937|25.3554|0|False|
|benchpress/bench press_40.mp4|0.0000|1.0028|N/A|67.1202|32.4241|66.3882|23.6195|27.7000|0|False|
|benchpress/stu10_11.mp4|70.4356|1.6432|108.4624|34.1230|27.6707|35.8233|15.7062|17.0100|247|True|
|benchpress/stu8_15.mp4|0.0000|1.6117|N/A|45.9273|37.2916|46.9968|22.1853|27.1430|0|False|
|benchpress/stu8_7.mp4|25.6541|2.5696|N/A|58.8982|38.3209|44.5104|31.7162|36.6207|0|False|
|benchpress/test153.mp4|76.0401|1.7812|N/A|40.2159|25.1914|34.8868|16.7573|23.2945|0|False|
|deadlift/deadlift_7.mp4|0.0000|2.7974|185.2715|50.8212|52.4956|50.2258|32.7619|28.6640|49|True|
|deadlift/deadlift_23.mp4|60.1548|2.9372|143.1803|42.2576|42.0016|41.4181|26.9302|22.3555|38|True|
|deadlift/deadlift_30.mp4|17.4679|2.6670|218.1213|59.9545|62.2046|58.6286|38.3215|34.1999|97|True|
|squat/stu6_11_2.mp4|0.0000|1.6064|160.2033|46.8820|46.4476|43.7299|28.0784|18.1913|59|True|
|squat/stu6_11_3.mp4|0.0000|1.8073|157.6373|46.0456|40.9714|44.1795|28.9169|24.9745|11|True|
|squat/val1250.mp4|63.4061|2.1658|164.5189|50.5512|47.8538|49.7882|28.0908|21.0676|87|True|
|benchpress/bench press_61.mp4|49.4662|2.0524|N/A|41.3743|38.1361|43.5635|20.1120|23.2420|0|False|
|benchpress/stu1_6_4.mp4|55.1632|2.5604|N/A|44.1655|30.8834|27.6952|19.6952|26.0104|0|False|
|benchpress/stu7_12_2.mp4|32.7935|1.5336|N/A|40.5377|42.1823|37.9209|28.2837|22.2054|0|False|
|benchpress/stu7_7.mp4|0.0000|2.3301|N/A|79.0033|40.7471|85.2898|31.1243|31.1167|0|False|
|benchpress/stu8_6_2.mp4|0.0000|0.7952|N/A|96.7279|28.2828|73.1150|33.6008|28.1845|0|False|
|deadlift/deadlift_sihoo02.mp4|34.0977|3.6288|174.8212|48.3044|50.1053|47.1775|32.4510|29.3715|70|True|
|deadlift/PB_S01_R10.mp4|0.0000|2.8685|225.6401|65.1820|63.3532|64.3544|42.8842|36.9983|66|True|
|deadlift/PB_S03_R13.mp4|4.8015|1.2847|165.5069|49.1554|46.4416|44.8300|29.0728|25.4483|42|True|
|deadlift/PB_S03_R14.mp4|13.8651|1.2646|163.7919|47.7602|45.5309|44.6709|28.6287|25.0127|42|True|
|deadlift/PB_S04_R03.mp4|9.6172|3.1066|238.1343|67.3662|65.7170|67.9997|44.3685|35.2570|52|True|
|deadlift/PB_S05_R22.mp4|0.0000|2.9387|270.6319|76.9551|73.8462|73.8152|48.7502|42.1236|42|True|
|deadlift/PB_S06_R25.mp4|7.6860|2.8417|223.7613|64.6711|63.0455|63.0233|41.3448|37.5238|41|True|
|deadlift/PB_S07_R19.mp4|0.0000|3.1072|265.5501|73.6133|73.8851|70.7815|47.1505|42.5622|57|True|
|deadlift/GB_S01_R01.mp4|0.0000|2.8091|247.1893|72.4137|69.7419|70.1696|45.1161|38.4801|54|True|
|deadlift/GB_S01_R07.mp4|0.0000|2.8515|261.6656|72.9157|71.1303|72.7116|46.9041|40.6922|61|True|
|deadlift/GB_S01_R11.mp4|16.5534|2.9890|243.4131|68.5530|66.9663|66.5250|42.2755|36.8996|60|True|
|deadlift/GB_S01_R16.mp4|0.0000|2.9149|252.6381|71.7541|69.4253|69.7103|44.7851|39.0151|66|True|
|deadlift/GB_S01_R18.mp4|0.0000|2.8771|247.7169|70.0656|67.5457|67.6122|43.0347|37.2477|63|True|
|deadlift/GB_S04_R03.mp4|0.0000|2.4267|184.5695|51.1047|50.6431|52.1878|35.3133|26.3934|39|True|
|deadlift/GB_S04_R04.mp4|2.3257|2.3865|188.5760|51.9204|51.0839|51.5078|34.2872|27.3008|54|True|
|deadlift/GB_S04_R07.mp4|14.1083|2.4125|189.1542|52.2744|51.4333|51.5306|34.0514|28.2380|45|True|
|deadlift/GB_S05_R47.mp4|0.0000|2.1498|204.9010|57.6587|55.6257|56.6661|37.6997|32.4324|40|True|
|deadlift/GB_S06_R63.mp4|0.0000|2.1961|186.1571|48.6224|51.9624|52.1583|35.9120|30.5211|27|True|
|deadlift/GB_S07_R12.mp4|25.0490|2.8598|266.0240|72.3472|70.4392|72.6847|46.6318|39.6587|63|True|
|deadlift/GB_S07_R14.mp4|0.0000|2.8858|269.5869|73.9351|71.1467|71.9716|46.8563|39.8953|67|True|
|deadlift/GB_S07_R20.mp4|39.5243|2.7716|248.9012|68.3351|65.4260|66.8505|43.3467|37.2072|61|True|
|deadlift/LLK_S01_R02.mp4|0.0000|2.5717|254.4480|72.1649|70.5523|72.0223|47.5158|40.1251|41|True|
|deadlift/LLK_S01_R03.mp4|0.0000|2.7574|240.1036|67.4766|66.9367|69.0120|46.8320|37.4609|43|True|
|deadlift/LLK_S01_R04.mp4|0.0000|2.5740|238.1150|69.3045|68.0327|68.1403|45.5144|37.6262|40|True|
|deadlift/LLK_S01_R06.mp4|0.0000|2.3174|249.2061|73.5668|66.3023|70.4124|42.0046|36.3704|40|True|
|deadlift/LLK_S03_R11.mp4|16.8747|1.2426|163.9111|47.8511|44.0828|46.0477|26.3501|24.4788|38|True|
|deadlift/LLK_S04_R05.mp4|0.0000|3.2910|237.6767|64.8079|66.4382|70.6245|49.1210|35.8077|33|True|
|deadlift/LLK_S05_R15.mp4|0.0000|3.1049|263.9759|72.4533|73.0464|74.8773|49.0442|40.1831|29|True|
|deadlift/LLK_S05_R17.mp4|0.0000|3.1032|268.0718|74.1952|74.8347|77.0532|50.7087|40.6781|30|True|
|deadlift/LLK_S06_R31.mp4|0.0000|2.8515|233.8733|64.9272|66.6191|65.1551|42.6410|36.8611|28|True|
|deadlift/LLK_S06_R32.mp4|0.0000|2.9219|236.8258|62.5642|66.7350|64.7398|43.7924|36.8100|28|True|
|deadlift/deadlift_01.mp4|50.6098|1.9573|152.9038|41.4125|43.3166|42.0859|29.7107|26.0053|51|True|
|deadlift/deadlift_02.mp4|50.9300|2.0197|152.4059|42.0416|43.8867|39.6635|28.0949|25.9429|60|True|
|deadlift/deadlift_03.mp4|52.3795|2.0078|141.3231|41.6566|42.5777|39.8491|27.5704|24.9280|206|True|
|benchpress/bench_01.mp4|41.7063|1.5891|N/A|85.8395|51.9715|80.5287|26.8111|33.2586|0|False|
|benchpress/bench_02.mp4|34.9821|1.8305|N/A|71.2213|44.4140|52.0847|25.6182|30.5014|0|False|
|benchpress/bench_03.mp4|39.1456|1.8125|N/A|74.6065|46.2116|54.8827|26.7207|35.7496|0|False|
|squat/squat_01.mp4|48.9703|1.8099|179.6107|53.7923|51.4625|49.6905|29.5394|26.0104|271|True|
|squat/squat_02.mp4|59.0943|3.0316|165.5199|48.7875|44.3065|51.2296|31.2485|15.9910|181|True|
|squat/squat_03.mp4|59.3059|2.7361|156.5598|46.6575|47.0327|44.5509|27.1879|15.6591|198|True|
|team9/squat/stu10_70.mp4|33.9917|1.7524|162.6578|45.4241|46.5762|44.1028|29.0583|24.3120|267|True|
|team9/squat/stu3_64.mp4|41.4759|1.9552|162.0734|48.6117|46.3992|44.7981|31.4203|19.4912|474|True|
|team9/squat/stu8_67.mp4|52.2497|1.4071|236.7731|77.0585|69.3227|59.6655|38.9393|23.0050|121|True|
|team9/benchpress/stu9_10.mp4|16.5570|3.0337|173.0819|58.9515|40.7965|56.0102|27.7002|29.8013|170|True|
|team9/benchpress/stu2_10_edited.mp4|40.4448|1.1623|N/A|61.3103|29.1482|54.4533|24.2109|24.7370|0|False|
|team9/deadlift/stu7_1_1.mp4|0.0000|1.8841|170.3318|45.9265|48.8873|49.5481|29.6346|23.5615|108|True|
|team9/deadlift/stu4_2_-1.mp4|48.0269|2.7014|194.8170|59.8769|63.1085|59.6007|39.4881|34.8153|238|True|
|team9/deadlift/deadlift_08.mp4|46.7698|2.0916|158.5575|43.9226|46.4864|42.9983|29.3076|26.1429|77|True|

키 보정 False인 분절은 모델 자체 미터 스케일이며 원판 키로 교정되지 않았다. 키 null은 추정 불가다.

### 키 이상치와 추정 불가

|영상|키 cm|바 간격 보정 전 cm|kd 중앙값|원인/가정 한계|
|---|---:|---:|---:|---|
|squat/val1235.mp4|143.6630|98.6128|1.4455|rep 안 오선택과 가려진 몸통 관측. 서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|benchpress/bench press_31.mp4|N/A|N/A|1.2157|No standing 3D frame: camera-up fallback; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|benchpress/bench press_40.mp4|N/A|N/A|1.6981|No standing 3D frame: camera-up fallback; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|benchpress/stu10_11.mp4|108.4624|94.5547|1.1426|완료 rep 없음; pre-lift 오선택과 편집/가림으로 standing 관측 오염 가능. 서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|benchpress/stu8_15.mp4|N/A|N/A|1.4343|No standing 3D frame: camera-up fallback; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|benchpress/stu8_7.mp4|N/A|N/A|1.2456|No standing 3D frame: camera-up fallback; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|benchpress/test153.mp4|N/A|N/A|1.0948|No standing 3D frame: camera-up fallback; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|deadlift/deadlift_23.mp4|143.1803|127.4555|1.1186|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/deadlift_30.mp4|218.1213|174.4446|1.2504|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|benchpress/bench press_61.mp4|N/A|N/A|1.2217|No standing 3D frame: camera-up fallback; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|benchpress/stu1_6_4.mp4|N/A|N/A|1.1562|No standing 3D frame: camera-up fallback; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|benchpress/stu7_12_2.mp4|N/A|N/A|1.3837|No standing 3D frame: camera-up fallback; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|benchpress/stu7_7.mp4|N/A|N/A|1.3004|No standing 3D frame: camera-up fallback; Toe heading sign unobservable/ambiguous; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|benchpress/stu8_6_2.mp4|N/A|N/A|1.8803|No standing 3D frame: camera-up fallback; Toe heading sign unobservable/ambiguous; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|deadlift/PB_S01_R10.mp4|225.6401|182.5403|1.2440|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/PB_S04_R03.mp4|238.1343|195.6128|1.2222|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/PB_S05_R22.mp4|270.6319|217.3754|1.2382|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/PB_S06_R25.mp4|223.7613|181.5227|1.2441|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/PB_S07_R19.mp4|265.5501|216.5765|1.2253|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/GB_S01_R01.mp4|247.1893|197.9960|1.2492|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/GB_S01_R07.mp4|261.6656|211.6486|1.2455|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/GB_S01_R11.mp4|243.4131|199.2443|1.2245|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/GB_S01_R16.mp4|252.6381|204.1429|1.2401|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/GB_S01_R18.mp4|247.7169|199.1083|1.2433|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/GB_S05_R47.mp4|204.9010|154.5476|1.3256|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/GB_S07_R12.mp4|266.0240|217.2153|1.2218|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/GB_S07_R14.mp4|269.5869|216.0724|1.2426|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/GB_S07_R20.mp4|248.9012|208.9527|1.1948|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/LLK_S01_R02.mp4|254.4480|200.8828|1.2722|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/LLK_S01_R03.mp4|240.1036|191.6647|1.2539|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/LLK_S01_R04.mp4|238.1150|188.5565|1.2720|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/LLK_S01_R06.mp4|249.2061|190.2360|1.3021|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/LLK_S04_R05.mp4|237.6767|195.1685|1.2127|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/LLK_S05_R15.mp4|263.9759|214.5817|1.2255|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/LLK_S05_R17.mp4|268.0718|218.7969|1.2256|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/LLK_S06_R31.mp4|233.8733|189.6186|1.2455|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/LLK_S06_R32.mp4|236.8258|190.8437|1.2396|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|deadlift/deadlift_03.mp4|141.3231|116.5782|1.2128|서 있는 2D 투영 높이와 원판 면/리프터 면 가정 편향; 실측 키 정답 없음|
|benchpress/bench_01.mp4|N/A|N/A|1.3289|No standing 3D frame: camera-up fallback; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|benchpress/bench_02.mp4|N/A|N/A|1.3133|No standing 3D frame: camera-up fallback; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|benchpress/bench_03.mp4|N/A|N/A|1.2995|No standing 3D frame: camera-up fallback; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|
|team9/squat/stu8_67.mp4|236.7731|181.1141|1.3046|발/신체 잘림과 원근·고정 초점/바 깊이 가정 편향; 실측 키 정답 없음|
|team9/benchpress/stu2_10_edited.mp4|N/A|N/A|1.4583|No standing 3D frame: camera-up fallback; Stature/standing unavailable: native VideoPose3D metre scale retained; No supported upright straight-knee standing frame for plate stature|

### 같은 사람으로 본 클립 비교

영상 확인에서 같은 얼굴·체형·장비/체육관을 가진 `squat_01/02/03`과 `deadlift_01/02/03`을 각각 한 그룹으로 봤다. 원본 데이터에 신원 정답은 없으므로 시각적 그룹 가정이다. 존재하지 않는 04/05를 추가하지 않았다.

|그룹|지표 cm|클립 수|최대−최소|모든 쌍 차이 중앙값|
|---|---|---:|---:|---:|
|squat_01/02/03|stature|3|23.0509|14.0908|
|squat_01/02/03|thigh|3|7.1347|5.0048|
|squat_01/02/03|shin|3|7.1560|4.4298|
|squat_01/02/03|torso|3|6.6787|5.1396|
|squat_01/02/03|upper_arm|3|4.0606|2.3515|
|squat_01/02/03|forearm|3|10.3513|10.0194|
|deadlift_01/02/03|stature|3|11.5806|11.0828|
|deadlift_01/02/03|thigh|3|0.6291|0.3850|
|deadlift_01/02/03|shin|3|1.3091|0.7389|
|deadlift_01/02/03|torso|3|2.4224|2.2368|
|deadlift_01/02/03|upper_arm|3|2.1403|1.6158|
|deadlift_01/02/03|forearm|3|1.0773|1.0149|

### 450 → 400 mm 예시

`squat/squat_01.mp4`의 같은 프레임에서 키는 179.6107 → 163.6901 cm (-15.9206 cm, 비율 0.911361)다. 원판 면 거리/깊이/속도 비율은 정확히 400/450=0.888889이지만 키에는 지름과 무관한 0.70m 항이 있어 그 비율과 다르다. 분절의 추가 스케일 비율은 같은 키 비율이다.

## 2D·3D rep 간 흔들림

영상별 SD(ddof=1)와 모든 쌍 절대 차이 중앙값을 구한 뒤 종목별 영상 중앙값을 낸다. 2D는 M1b n10 원래 값이다. 벤치에는 guard 변경도 섞여 있으므로 차원을 바꾼 효과만으로 해석하지 않는다.

|범위|포지션|지표|2D n|2D SD|2D 쌍 차이|3D n|3D SD|3D 쌍 차이|
|---|---|---|---:|---:|---:|---:|---:|---:|
|all|START|knee_angle|16|5.5213|7.0043|20|6.7886|7.4334|
|all|START|hip_angle|19|4.2310|5.0216|20|3.7123|4.5291|
|all|START|trunk_lean|19|1.6471|1.7409|20|1.4311|1.6828|
|all|MID_DOWN|knee_angle|15|3.0204|4.2715|20|3.5733|4.5321|
|all|MID_DOWN|hip_angle|18|3.0087|3.9074|20|3.2307|4.5198|
|all|MID_DOWN|trunk_lean|18|2.0516|2.4632|20|1.7171|2.2825|
|all|BOTTOM|knee_angle|15|3.6693|4.6098|20|3.5162|4.5870|
|all|BOTTOM|hip_angle|15|3.4075|4.5481|20|3.6801|4.5828|
|all|BOTTOM|trunk_lean|15|2.2071|2.4554|20|2.0374|2.7577|
|all|MID_UP|knee_angle|14|3.2699|4.6244|20|6.2467|7.2331|
|all|MID_UP|hip_angle|17|4.3263|6.1182|20|3.3280|4.0670|
|all|MID_UP|trunk_lean|16|2.6740|2.9961|20|2.3815|3.3264|
|all|LOCKOUT|knee_angle|17|6.5562|9.2718|20|5.9183|8.3759|
|all|LOCKOUT|hip_angle|20|5.9601|8.2107|20|3.0473|3.9521|
|all|LOCKOUT|trunk_lean|20|3.2072|3.9734|20|2.0166|2.5463|
|squat|START|knee_angle|5|0.5119|0.5412|6|2.8509|3.8351|
|squat|START|hip_angle|5|1.8622|1.8763|6|3.7123|4.5833|
|squat|START|trunk_lean|5|1.2578|1.3045|6|0.7999|1.0038|
|squat|MID_DOWN|knee_angle|5|2.1460|3.6145|6|3.5733|4.6272|
|squat|MID_DOWN|hip_angle|5|2.4242|2.9669|6|3.3573|4.2929|
|squat|MID_DOWN|trunk_lean|5|0.9534|1.3483|6|1.1589|1.7694|
|squat|BOTTOM|knee_angle|5|3.6693|4.6098|6|3.5162|4.9079|
|squat|BOTTOM|hip_angle|5|3.9626|4.8190|6|2.7721|3.4862|
|squat|BOTTOM|trunk_lean|5|0.4081|0.2965|6|2.3722|2.5053|
|squat|MID_UP|knee_angle|4|2.3631|2.1474|6|4.9038|6.8706|
|squat|MID_UP|hip_angle|5|1.5172|2.2053|6|5.0371|5.0544|
|squat|MID_UP|trunk_lean|5|0.8229|1.1637|6|1.1373|1.5214|
|squat|LOCKOUT|knee_angle|5|4.2835|4.1678|6|2.0688|2.4189|
|squat|LOCKOUT|hip_angle|6|7.7752|9.7991|6|3.7227|3.9977|
|squat|LOCKOUT|trunk_lean|6|2.1595|3.0541|6|1.5534|1.8402|
|benchpress|START|knee_angle|7|14.9460|10.1165|10|10.1996|11.2501|
|benchpress|START|hip_angle|10|4.7778|7.3395|10|3.6913|4.4390|
|benchpress|START|trunk_lean|10|1.2827|1.7401|10|1.3599|1.7040|
|benchpress|MID_DOWN|knee_angle|6|6.9984|9.8972|10|4.5613|6.7523|
|benchpress|MID_DOWN|hip_angle|9|2.7871|3.9415|10|3.5947|4.7940|
|benchpress|MID_DOWN|trunk_lean|9|2.2931|3.2429|10|2.2722|2.7983|
|benchpress|BOTTOM|knee_angle|6|12.0596|17.4235|10|7.2843|8.6148|
|benchpress|BOTTOM|hip_angle|6|2.4869|3.8481|10|4.2916|5.7551|
|benchpress|BOTTOM|trunk_lean|6|2.5855|3.5460|10|2.8367|3.7837|
|benchpress|MID_UP|knee_angle|6|10.6862|12.3944|10|17.2607|18.8966|
|benchpress|MID_UP|hip_angle|8|10.4771|14.0136|10|3.0936|4.9662|
|benchpress|MID_UP|trunk_lean|7|6.2200|8.7083|10|3.2302|4.9705|
|benchpress|LOCKOUT|knee_angle|8|5.9672|8.6678|10|7.7294|9.9039|
|benchpress|LOCKOUT|hip_angle|10|2.3763|3.1574|10|2.2998|2.8212|
|benchpress|LOCKOUT|trunk_lean|10|1.5748|2.1532|10|1.9234|2.5463|
|deadlift|START|knee_angle|4|6.6571|8.4662|4|7.7053|8.7329|
|deadlift|START|hip_angle|4|5.4730|7.5087|4|3.9819|5.7341|
|deadlift|START|trunk_lean|4|2.5829|3.7605|4|2.2881|3.0307|
|deadlift|MID_DOWN|knee_angle|4|1.4470|1.9077|4|1.7351|2.2029|
|deadlift|MID_DOWN|hip_angle|4|3.4853|4.5815|4|3.2029|4.5296|
|deadlift|MID_DOWN|trunk_lean|4|2.7618|3.3099|4|2.1202|2.4438|
|deadlift|BOTTOM|knee_angle|4|1.1775|1.7148|4|1.4317|1.7380|
|deadlift|BOTTOM|hip_angle|4|3.0939|4.2158|4|2.6339|3.2707|
|deadlift|BOTTOM|trunk_lean|4|1.3825|1.4850|4|1.9570|2.6439|
|deadlift|MID_UP|knee_angle|4|2.4331|3.0967|4|3.1867|4.1052|
|deadlift|MID_UP|hip_angle|4|5.2700|6.7780|4|2.9918|1.7561|
|deadlift|MID_UP|trunk_lean|4|3.0423|4.1035|4|2.3815|3.2708|
|deadlift|LOCKOUT|knee_angle|4|7.5436|9.7171|4|6.3465|7.3739|
|deadlift|LOCKOUT|hip_angle|4|10.5931|14.1316|4|6.2252|8.3061|
|deadlift|LOCKOUT|trunk_lean|4|7.2309|10.2260|4|2.6056|3.5411|

## 3D 이미지 이상 목록

|영상|프레임|관찰|이미지|
|---|---:|---|---|
|deadlift/GB_S04_R03.mp4|3|Large wrist height disagreement at common bar and floating far ankle in 3D side view|pose3d/images/deadlift__GB_S04_R03.mp4.jpg|
|deadlift/LLK_S01_R02.mp4|129|Far ankle floats; projected arm and shins cross; 3D depth/left-right assignment uncertain|pose3d/images/deadlift__LLK_S01_R02.mp4.jpg|
|deadlift/LLK_S01_R03.mp4|145|Far ankle floats; projected arm and shins cross despite planted feet|pose3d/images/deadlift__LLK_S01_R03.mp4.jpg|
|deadlift/LLK_S01_R04.mp4|1|Far ankle floats; projected arm and shins cross despite planted feet|pose3d/images/deadlift__LLK_S01_R04.mp4.jpg|
|deadlift/LLK_S01_R06.mp4|119|Far ankle floats and strong leg-depth asymmetry|pose3d/images/deadlift__LLK_S01_R06.mp4.jpg|
|deadlift/LLK_S05_R15.mp4|3|3D wrist/forearm projects through knee; arm depth suspect|pose3d/images/deadlift__LLK_S05_R15.mp4.jpg|
|deadlift/LLK_S05_R17.mp4|78|3D wrist/forearm projects through knee; arm depth suspect|pose3d/images/deadlift__LLK_S05_R17.mp4.jpg|
|deadlift/LLK_S06_R31.mp4|99|Large wrist height imbalance at common bar; projected arm crosses leg|pose3d/images/deadlift__LLK_S06_R31.mp4.jpg|
|deadlift/deadlift_23.mp4|36|3D knees visibly flex despite near-upright 2D; uncertain leg depth, no completed rep|pose3d/images/deadlift__deadlift_23.mp4.jpg|
|deadlift/deadlift_sihoo02.mp4|115|Large 3D wrist height disagreement; source lower body is small/occluded by overlay|pose3d/images/deadlift__deadlift_sihoo02.mp4.jpg|
|squat/squat_02.mp4|187|Large left/right foot height split in side projection despite both feet planted in source; depth/ankle reconstruction suspect|pose3d/images/squat__squat_02.mp4.jpg|
|squat/val1235.mp4|81|Asymmetric legs and trunk; input has known wrong-lifter samples in rep, 3D identity is not independently corrected|pose3d/images/squat__val1235.mp4.jpg|
|squat/val1256.mp4|106|Source upper body and hands occluded/cropped; 3D elbows/wrists extrapolated; exact handedness not verifiable; far ankle floats in side view|pose3d/images/squat__val1256.mp4.jpg|
|team9/squat/stu10_70.mp4|56|Large left/right foot height split and arm overlap in side view; depth/foot reconstruction suspect|pose3d/images/team9__squat__stu10_70.mp4.jpg|
|team9/squat/stu3_64.mp4|112|One ankle floats substantially above floor; asymmetric depth and source text occlusion|pose3d/images/team9__squat__stu3_64.mp4.jpg|
|team9/squat/stu8_67.mp4|67|Cropped legs and elevated reconstructed ankle; geometry stature is implausible (>200 cm)|pose3d/images/team9__squat__stu8_67.mp4.jpg|
|benchpress/bench press_31.mp4|42|Lying lifter reconstructed with near-upright torso, strongly asymmetric arms/feet|pose3d/images/benchpress__bench press_31.mp4.jpg|
|benchpress/stu10_11.mp4|899|Very asymmetric arms and feet; no completed rep diagnostic; a pre-lift wrong-person interval can contaminate clip geometry|pose3d/images/benchpress__stu10_11.mp4.jpg|
|benchpress/stu8_15.mp4|137|Large projected leg crossing and wrist height asymmetry; body orientation/hidden legs unreliable|pose3d/images/benchpress__stu8_15.mp4.jpg|
|benchpress/stu8_7.mp4|301|Strongly asymmetric/crossed projected limbs and inclined upper body in lying source|pose3d/images/benchpress__stu8_7.mp4.jpg|
|benchpress/test153.mp4|150|Lying lifter reconstructed nearly upright with crossed projected legs; no completed rep diagnostic|pose3d/images/benchpress__test153.mp4.jpg|
|benchpress/bench press_61.mp4|25|Large left/right wrist height split and floating far ankle despite shared bar/ground|pose3d/images/benchpress__bench press_61.mp4.jpg|
|benchpress/stu1_6_4.mp4|75|Lying lifter reconstructed with inclined torso and crossed projected legs; feet from frontal view make depth uncertain|pose3d/images/benchpress__stu1_6_4.mp4.jpg|
|benchpress/stu7_12_2.mp4|95|Collapsed/crossed legs and pelvis below ankle-defined floor; lower body mostly absent from source|pose3d/images/benchpress__stu7_12_2.mp4.jpg|
|benchpress/stu7_7.mp4|42|Torso horizontal but elbow below ankle-defined floor and asymmetric wrist heights; occluded lower body extrapolated|pose3d/images/benchpress__stu7_7.mp4.jpg|
|benchpress/stu8_6_2.mp4|25|Predicted arms flatten along torso despite source vertical forearm; occluded hand/leg depth unreliable|pose3d/images/benchpress__stu8_6_2.mp4.jpg|
|benchpress/bench_01.mp4|262|Torso roughly horizontal but reconstructed wrists drop below torso instead of staying at shared bar; arm directions suspect|pose3d/images/benchpress__bench_01.mp4.jpg|
|benchpress/bench_02.mp4|265|Torso overly inclined and wrists drop below body relative to source bench posture|pose3d/images/benchpress__bench_02.mp4.jpg|
|benchpress/bench_03.mp4|223|Torso overly inclined; strong left/right wrist height split at shared bar|pose3d/images/benchpress__bench_03.mp4.jpg|
|team9/benchpress/stu9_10.mp4|263|Lying lifter reconstructed almost upright with crossed legs and wrists; native scale/standing geometry contaminated by clip-wide posture changes|pose3d/images/team9__benchpress__stu9_10.mp4.jpg|
|team9/benchpress/stu2_10_edited.mp4|43|Lying lifter reconstructed nearly upright with projected arm/leg crossing|pose3d/images/team9__benchpress__stu2_10_edited.mp4.jpg|

전체 69개 이미지의 검토·입력 해시는 `pose3d/image_review.json`에 있다. 뒤집힘/좌우 바뀜을 확정할 3D 정답은 없으며, 명백한 투영 이상과 불확실성을 구분했다.

## 속도

|범위|프레임|추론 초|기하·후처리 초|합계 초|
|---|---:|---:|---:|---:|
|all|19331|15.8112|0.2967|16.1079|
|squat|4781|3.8249|0.1272|3.9521|
|benchpress|8448|6.4350|0.0716|6.5066|
|deadlift|6102|5.5512|0.0980|5.6492|

|설정, ORT 6|원판+2D 초|3D 추론 초|3D 기하/후처리 초|모델 준비·저장 포함 처리 초|지표 포함 전체 초|
|---|---:|---:|---:|---:|---:|---:|
|M1b n10 (당시 guard on)|125.3975|N/A|N/A|125.3975|125.9779|
|M2 n10 guard off + 3D|121.5461|0.4945|0.0110|122.8580|123.4226|

동일 입력 SHA256: `cc1d4eb5ccc63ac4bea6aa7754fc960dd698956ad79c6715d8f185639f208047`. M2의 torch도 6스레드다. 준비된 60초/30fps 영상을 단독 처리했다.

## 구현·가정·재현


기준 `plate-pipeline(d5e19be)`, 브랜치 `integ/m2-3d`. PLAN.md의 M1b 결정을 읽었고 PLAN.md/main은 수정하지 않는다. 모델·영상·eval_out은 커밋하지 않는다.

## 포즈와 입력

PoseCfg 기본값은 tracked, det_frequency=10, bench_rep_guard=False다. 벤치 16개는 이 설정으로 원판·포즈부터 다시 추론한다. 나머지 53개는 M1b n10의 선택된 raw MP33를 재사용한다. guard는 벤치에만 영향을 주므로 나머지는 같은 선택 경로다. VideoPose3D 입력은 이 raw 관절에서 COCO17을 가져오며 사람 검출이나 최대 상자 선택을 다시 하지 않는다. visibility가 낮다고 다른 사람 관절로 대체하지 않는다.

기존 검토 문구는 원본 그대로 보존하고, 잘못된 사람으로 확인한 표본 프레임을 완료 원판 rep 구간 안/밖으로 분류했다. 과거 문구에서 "고침"으로 나오는 프레임까지 오류로 세지 않는다. 완료 rep은 start/far_in/far_out/back이 모두 있는 경우이며 start~back 양 끝을 포함한다. 한 영상이 두 범주에 모두 나타날 수 있다. 이미지 표본을 기준으로 한 하한이며 전체 프레임의 정답 판정률이 아니다.

벤치 재평가는 ORT intra-op 1/inter-op 1, OpenCV 1이다. 영상 묶음을 네 개로 나눠 서로 다른 물리 코어 CPU 0/3/4/5에서 실행했다. 3D의 최종 69개 측정은 CPU 1에 고정해 1스레드로 순차 실행했다. 설정별 영상은 순차 처리했다. 6스레드 전체 처리 시간은 이 작업과 시험이 모두 끝난 뒤 단독 실행한다.

## 기하

`core.geometry.plate_geometry`는 사용된 원판 상자의 가로/세로 비의 중앙값을 [0,1]로 잘라 arccos로 각도를 구한다. 초점거리는 `0.77 × max(w,h)`, 거리는 `f × D / 상자 세로`, kd는 `(거리 + 0.70 × cos 각도) / 거리`다. 상자가 없는 프레임의 세로는 양쪽 관측값으로 선형 보간하고 짧은 끝 구간은 가까운 관측값을 쓴다. 상자 관측 수와 비율 원값도 저장한다. 원판 신호나 판정 코드는 바꾸지 않는다.

0.77은 초점거리 가정, 0.70m는 **바 기하 가정, 지름과 무관**, 0.92는 코–바닥 높이/키 비율 가정이다. 모두 GeometryCfg에 있다. 사용자의 추가 결정대로 시험을 나눴다. 원판 면의 거리·깊이·속도는 지름에 비례한다. 리프터 면 스케일은 `D/h + 0.70*cos(angle)/f`이며 고정 바 간격 때문에 지름에 정확히 비례하지 않는다. 보고서에 실제 영상의 450→400mm 예시를 넣는다.

서 있는 프레임은 선택된 리프터의 코·양 어깨·엉덩이·무릎·발목이 유한하고 visibility≥0.5, 양 무릎 각도>160°, 몸통이 화면 수직에서 30° 미만이며 어깨가 엉덩이 위인 프레임이다. 가시 발목·뒤꿈치·발끝 중 가장 아래를 바닥으로 보고 `(발 y−코 y) × 리프터 면 m/px / 0.92`의 중앙값을 키로 쓴다. 누운 벤치 영상에서 다른 선 사람을 키 측정 대상으로 다시 고르지 않는다. 서 있는 관측이 없으면 키는 null이고 원인을 기록한다. 숫자를 150~200cm로 강제로 자르지 않는다.

## 3D

`track3d`의 vendored VideoPose3D와 `pretrained_h36m_detectron_coco.bin`을 재사용한다. 수용 영역은 243프레임이다. track3d의 프레임 단위 최대 10프레임 빈칸 선형 보간·짧은 끝 구간 hold를 그대로 쓰며, 긴 빈칸은 null로 남기고 유효 구간을 분리한다. 추론은 128 출력 프레임씩 실행하되 앞뒤 121프레임 문맥을 보존한다. 작은 합성 TemporalModel에서 전체 구간 추론과 수치 일치를 시험한다.

Gaussian 시간 평활화는 sigma=max(1프레임, fps×0.06초), 반경 ceil(3sigma), 끝은 nearest다. 긴 빈칸을 가로질러 평활화하지 않는다. H36M tree의 16개 뼈를 각 세트의 평활화 후 길이 중앙값으로 고정한다. 각 뼈 방향을 유지하며 부모부터 재구축한다. rep 밖 프레임은 가장 가까운 세트 중심에 배정하며 완료 세트가 없으면 영상 전체를 한 그룹으로 쓴다. 서로 다른 세트의 목표 길이는 다를 수 있으므로 CV는 세트별로도 저장한다.

몸 좌표계 위쪽은 서 있는 3D 프레임의 발목 중점→neck/nose(9) 벡터 중앙값이다. 엉덩이 좌우선을 위쪽에 직교화해 lateral 축을 만들고 그 선과 위쪽의 외적으로 forward 축을 만든다. Halpe 발뒤꿈치→엄지발가락의 2D 방향과 forward의 영상 평면 투영 내적으로 부호를 정한다. 발 방향이 가려지거나 수직 투영이 0에 가까우면 부호 불확실을 기록한다. 서 있는 프레임이 없으면 camera-up [0,-1,0]을 사용했다는 경고를 남긴다.

측정 키의 0.92배와 3D neck/nose→낮은 발목 높이 중앙값이 같도록 스케일한다. 서 있는 관측/키가 없으면 **VideoPose3D의 모델 자체 미터 스케일**을 유지하고 `scaled_to_stature=false`로 명시한다. 이 경우 키로 교정된 분절 수치라고 해석하면 안 된다. 프레임마다 낮은 발목 y를 0으로 옮기고 서 있는 발목 중점의 x/z 중앙값을 원점으로 둔다. 양발을 동시에 강제로 바닥에 붙이지 않는다.

출력은 프레임별 H36M17 미터 좌표와 MP33 번호 테이블(`j3names`, `j3`)이다. H36M 9는 기존 track3d에서 neck라 부르는 neck/nose proxy다. 독립적인 COCO 코가 없으므로 MP0에 그 proxy를 넣었다고 명시한다. 없는 MP 관절은 생성하지 않는다. 무릎·엉덩이·몸통 기울기는 기존 다섯 포지션에서 계산해 별도 JSON에 저장하며 기존 판정에는 넣지 않는다.

확인 이미지는 첫 완료 rep에서 원판 중심 y가 가장 큰 **물리적으로 가장 낮은 원판 프레임**을 쓴다. 데드리프트는 far/top과 구분한다. 완료 rep이 없으면 영상 중간 프레임에 NO COMPLETED REP 진단 표시를 넣는다. 왼쪽은 기존 raw 2D, 오른쪽은 z-forward/y-up의 등방 측면 투영이다. 파랑은 왼쪽, 주황은 오른쪽이다. 측면 투영만으로 숨은 깊이나 좌우 정답을 확정할 수 없다.

분절은 보정한 3D의 양쪽 길이 중앙값이다. H36M hip→shoulder 몸통은 여러 tree 뼈를 가로지르므로 그 합성 선 길이까지 불변이라는 뜻은 아니다. 뼈 CV는 평활화 후 고정 전/후 비교다. 가속도는 같은 최종 스케일을 적용한 native/평활화 좌표의 2차 차분 norm×fps² 중앙값(m/s²)이며 바닥 고정으로 생기는 이동은 제외한다. 각도 흔들림은 M1b와 같은 ddof=1 SD·모든 rep 쌍 차이 중앙값이고 2D/3D 각각의 관측 수를 표시한다. 흔들림 감소가 3D 정답률 향상을 보증하지 않는다.

## 라이선스와 실행

VideoPose3D의 코드·가중치는 [공식 저장소](https://github.com/facebookresearch/VideoPose3D)와 vendored LICENSE의 **CC BY-NC 4.0** 조건을 따른다. COCO 입력용 가중치는 [공식 inference 안내](https://github.com/facebookresearch/VideoPose3D/blob/main/INFERENCE.md)의 파일이다. 상업적 사용 라이선스로 해석하지 않는다. 체크포인트는 저장소 밖 `~/.cache/videopose3d/`에 둔다.

```bash
cd plate_pipeline
python -m eval.baseline --out eval_out/m2/bench_n10 --mode tracked --det-frequency 10 \
  --no-bench-rep-guard --exercise-filter benchpress --prepared-reference M1_BALANCED_DIR --resume
python -m eval.m2_3d --source M1B_N10_DIR --bench eval_out/m2/bench_n10 \
  --out eval_out/m2/pose3d --checkpoint ~/.cache/videopose3d/pretrained_h36m_detectron_coco.bin --resume
# 실제 제품 경로의 선택된 pose로 동일한 진단 JSON 생성 (기존 판정에는 미사용)
cd ..
python run_demo.py --config CONFIG.json --segment plate --out OUT \
  --pose3d-checkpoint ~/.cache/videopose3d/pretrained_h36m_detectron_coco.bin
```

합성 시험은 기하 공식·두 면의 지름 법칙·뼈 고정과 방향 보존·회전 불변 관절각/몸통 기울기·크기/바닥·MP33/COCO 대응·짧은 보간/긴 빈칸·분할 추론·포즈 기본값을 다룬다. 확인 이미지와 수치 결과는 eval_out에 있고 최종 보고서만 results에 커밋한다.


## 측정 환경과 시험

AMD Ryzen 5 7500F, CPUExecutionProvider, rtmlib 0.0.16, onnxruntime 1.31.0,
Torch 2.14.1+cpu, SciPy 1.18.1, NumPy 2.5.3이다. 체크포인트 SHA256은
`d3219e005b50591f694da5cbaf6849f060d6b2cf895864a779f8a992ac63a232`다.
69개 표의 3D 시간은 배열 입력부터 추론·기하·후처리까지이며 일회성 모델 로드와
JSON/이미지 인코딩은 제외한다. 6스레드 전체 시간에는 모델 로드·저장·지표가 들어간다.

```bash
cd plate_pipeline
python -m pytest -q tests eval/midterm/test_metrics.py
# 160 passed in 10.88s
cd ..
PATH=/home/kojy1009/miniconda3/lib/python3.14/site-packages/playwright/driver:$PATH python -m pytest -q tests/fb
# 235 passed in 2.93s
```


### H36M → 데모 MP33 대응

|MP33|H36M17|의미|
|---|---|---|
|0|9|neck/nose proxy (독립 코 아님)|
|11, 12|11, 14|왼쪽·오른쪽 어깨|
|13, 14|12, 15|왼쪽·오른쪽 팔꿈치|
|15, 16|13, 16|왼쪽·오른쪽 손목|
|23, 24|4, 1|왼쪽·오른쪽 엉덩이|
|25, 26|5, 2|왼쪽·오른쪽 무릎|
|27, 28|6, 3|왼쪽·오른쪽 발목|

그 외 MP33 점은 null이며 발뒤꿈치/발끝의 3D를 만들어내지 않는다. 발 방향의 부호 판정에만 기존 Halpe 2D 발점을 쓴다.


## 보고서 재생성

```bash
cd plate_pipeline
python -m eval.m2_reviews --m0 M0_DIR --m1 M1_ROOT --m1b M1B_ROOT \
  --out eval_out/m2/lifter_review_intervals.json
python -m eval.m2_benchmark --video M1B_ROOT/benchmark_60s30.mp4 \
  --model PLATE_MODEL.onnx --checkpoint ~/.cache/videopose3d/pretrained_h36m_detectron_coco.bin \
  --out eval_out/m2/threads6
python -m eval.m2_report --m0 M0_DIR --m1 M1_ROOT --m1b M1B_ROOT \
  --runs eval_out/m2 --notes eval/M2_NOTES.md --report results/m2-3d/REPORT.md
```

실제 벤치 네 묶음의 명령·프레임별 측정 fingerprint는 `bench_n10/shards.json`에 있다.
그 결과를 같은 16개 순서로 모아 baseline을 만들었으며 `wall_s`는 네 명령 중 가장 긴
실행 시간이다. 병렬 묶음의 합성 fps로 속도 향상을 주장하지 않는다. 모든 입력은
M0/M1과 SHA256이 같은 준비 영상을 사용했고, 원판과 포즈는 새로 추론했다.
3D 단계는 최종 코드로 69개 모두 다시 생성했고 이미지 해시를 검토 기록과 대조한다.

zip은 설정별 JSON·MD와 이미지, 역사적 오류 표본의 안/밖 분류, 원판 불변 증거를 담는다.
모델·영상·NPZ는 zip에 넣지 않는다. 벤치에서 몸통/발이 화면 밖인 경우와 서 있는
프레임이 없는 경우를 3D 모델이 자동으로 해결했다고 해석하지 않는다.

3D 각도는 해당 17점 프레임이 유한하면 계산한다. track3d와 같이 낮은 2D 점수에
별도 confidence pruning을 추가하지 않았고, 짧은 빈칸 보간 프레임도 포함한다.
따라서 기존 관절별 visibility gate가 있는 2D보다 가용 rep 수가 많을 수 있다.
이 차이는 표의 n과 보간 인덱스로 드러내며 추가 관측이나 정확도 향상으로 보지 않는다.
6스레드 전체 시간의 M1b 대비 작은 차이는 실행 편차도 포함하므로 3D를 더해서
2D가 빨라졌다고 해석하지 않는다.
