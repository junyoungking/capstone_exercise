# M3c 화면 작업 보고

기준 브랜치: `integ/m3b-server` (`da471cf`); 작업 브랜치: `integ/m3c-ui`. M3b 브랜치는 origin에 push했다. 제품 변경은 `server/web/app.html`, `server/web/template3.html` 두 파일뿐이며 API, 분석, 수치, 저장 구조는 변경하지 않았다.

## 바뀐 화면

- 새 기록: 두 영상의 드롭 영역과 미리보기, 운동 선택, 원판 지름 및 선택 입력. 기준 영상은 새 업로드·저장된 영상·기준 없이 기록 중 선택한다. 512 MiB 초과 파일은 전송 전에 안내한다.
- 처리: 실제 업로드 영상 미리보기와 스캔 선, API의 단계별 상태·진행률·프레임 수·경과 시간과 전체 진행률. 실패와 재시도를 같은 흐름에 배치했다.
- 운동 기록: 실제 세션으로 채운 달력과 월간 요약, 날짜 선택 및 기록 목록. 세션이 없을 때는 빈 화면을 보인다.
- 기준 없는 기록 리포트: 실제 KPI와 rep 속도 막대, rep 표, 바 높이와 템포, 영상 및 3D/2D를 재배치했다. 마지막 rep 막대는 라임색이다.
- 상단 탐색과 두 해상도 레이아웃을 정리했다.

## v3 기준과 다른 점

데모에는 없던 저장 기준 영상 선택과 기준 없는 기록, 키 입력은 실제 기능을 위해 추가했다. 자동 인식은 제공하지 않는다. 처리 화면의 값은 API가 반환한 실제 값이며, 데모의 시간 연출이나 예시 기록을 제품에 넣지 않았다. 운동 기록 요약은 실제 세션의 종목별 최고 속도를 사용한다. 지난 기록 비교 이미지에서 데모는 여러 예시 세트를 보여 주고 제품은 실제 단일 세트를 보여 주므로 막대와 밀도가 다르다. 기준 캡처에만 데모의 목업 데이터가 포함된다.

## 실제 흐름과 검증

1920×1080 및 1440×900 브라우저에서 빈 기록 → 스쿼트 비교 → 기록 확인과 다시 열기 → 기준 없는 데드리프트 → 저장 기준 재사용을 수행했다. 스쿼트는 `stu8_67.mp4`와 `stu10_70.mp4`, 데드리프트는 `deadlift_08.mp4`를 사용했다. 키 175 cm는 화면 검증용 입력이다. 세 분석 모두 3회로 나왔고, 저장 기준 재사용에서 실제 캐시 적중을 확인했다. 드래그 앤 드롭, 직접 지름 입력, 크기 제한, 실패 및 재시도도 확인했다. 브라우저 콘솔 오류는 0개였다.

동일 입력의 M3b 결과와 비교하여 실행 시간과 캐시 메타데이터를 제외한 모든 분석 필드가 세 작업에서 정확히 일치했다 (`json_parity.json`). 캡처 시 가로 넘침은 없었다 (`layout_checks.json`, `final_checks.json`). 리포트는 세로로 스크롤한다. 원본 비율을 유지하는 영상에는 검은 여백이 남고, 3D의 발이 떠 보이는 기존 모델 표현도 관찰된다. 단일 세트 속도 막대는 데모의 여러 세트 막대보다 넓다.

- `cd plate_pipeline && python -m pytest -q tests eval/midterm/test_metrics.py`: 181 passed, Starlette 의존성 경고 1개.
- 루트에서 `python -m pytest -q tests/fb`: 235 passed. Playwright에 포함된 Node를 PATH에 추가하여 실행했다.

## 남은 제약

기존 API는 저장 기준 원본을 직접 제공하지 않는다. 해당 기준으로 완료된 세션이 있으면 그 세션의 영상을 미리보기로 쓰고, 완료 세션이 없으면 미리보기 불가를 안내하면서 선택은 허용한다. 브라우저에서 재생할 수 없는 코덱도 미리보기가 불가능하다. 진행 중 페이지를 새로 고치면 로컬 파일의 브라우저 미리보기는 복원되지 않는다. 이 제약을 없애려면 API나 저장 경로 변경이 필요하다.

## 캡처 목록

`reference/`는 v3 원본 데모이고, `product/`는 실제 제품 흐름이다. 파일명 끝의 해상도는 캡처 크기다. `reference/`에서 해상도 접미사가 없는 파일은 1920×1080이다. `comparisons/`는 왼쪽 데모, 오른쪽 제품의 나란한 비교이며 `comparison_manifest.json`에 각 원본을 기록했다.

### v3 기준

- `reference/v3_history.png`
- `reference/v3_history_1440x900.png`
- `reference/v3_past.png`
- `reference/v3_past_1440x900.png`
- `reference/v3_processing.png`
- `reference/v3_processing_1440x900.png`
- `reference/v3_report.png`
- `reference/v3_report_1440x900.png`
- `reference/v3_upload.png`
- `reference/v3_upload_1440x900.png`
- `reference/v3_upload_selected.png`
- `reference/v3_upload_selected_1440x900.png`

### 제품 화면

- `product/01_empty_1440x900.png`
- `product/01_empty_1920x1080.png`
- `product/02_upload_1440x900.png`
- `product/02_upload_1920x1080.png`
- `product/03_squat_cards_1440x900.png`
- `product/03_squat_cards_1920x1080.png`
- `product/03_squat_input_1440x900.png`
- `product/03_squat_input_1920x1080.png`
- `product/03_squat_processing_1440x900.png`
- `product/03_squat_processing_1920x1080.png`
- `product/03_squat_report_1440x900.png`
- `product/03_squat_report_1920x1080.png`
- `product/06_history_one_1440x900.png`
- `product/06_history_one_1920x1080.png`
- `product/07_day_selected_1440x900.png`
- `product/07_day_selected_1920x1080.png`
- `product/08_reopened_1440x900.png`
- `product/08_reopened_1920x1080.png`
- `product/09_deadlift_input_1440x900.png`
- `product/09_deadlift_input_1920x1080.png`
- `product/09_deadlift_processing_1440x900.png`
- `product/09_deadlift_processing_1920x1080.png`
- `product/09_deadlift_report_1440x900.png`
- `product/09_deadlift_report_1920x1080.png`
- `product/11_record_charts_1440x900.png`
- `product/11_record_charts_1920x1080.png`
- `product/12_record_media_1440x900.png`
- `product/12_record_media_1920x1080.png`
- `product/13_record_details_1440x900.png`
- `product/13_record_details_1920x1080.png`
- `product/14_saved_ref_cards_1440x900.png`
- `product/14_saved_ref_cards_1920x1080.png`
- `product/14_saved_ref_input_1440x900.png`
- `product/14_saved_ref_input_1920x1080.png`
- `product/14_saved_ref_processing_1440x900.png`
- `product/14_saved_ref_processing_1920x1080.png`
- `product/14_saved_ref_report_1440x900.png`
- `product/14_saved_ref_report_1920x1080.png`
- `product/17_history_three_1440x900.png`
- `product/17_history_three_1920x1080.png`
- `product/18_custom_diameter_1440x900.png`
- `product/18_custom_diameter_1920x1080.png`
- `product/19_failure_1440x900.png`
- `product/19_failure_1920x1080.png`
- `product/20_retry_1440x900.png`
- `product/20_retry_1920x1080.png`

### 나란히 비교

- `comparisons/comparison_report_1440x900.png`
- `comparisons/comparison_report_1920x1080.png`
- `comparisons/history_1440x900.png`
- `comparisons/history_1920x1080.png`
- `comparisons/processing_1440x900.png`
- `comparisons/processing_1920x1080.png`
- `comparisons/record_report_1440x900.png`
- `comparisons/record_report_1920x1080.png`
- `comparisons/selected_1440x900.png`
- `comparisons/selected_1920x1080.png`
- `comparisons/upload_1440x900.png`
- `comparisons/upload_1920x1080.png`
