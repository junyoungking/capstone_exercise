"""
feedback_integration_patch.py
==============================
realtime_compare_v1_1.py에 FeedbackRenderer를 연동하는 방법.

변경사항은 최소화. 기존 draw_user_skeleton, draw_expert_skeleton은
유지하고, 그 위에 FeedbackRenderer를 추가로 호출한다.

[적용 방법]
1. 파일 상단 import에 추가:
   from feedback_overlay import FeedbackRenderer, extract_expert_depth_target

2. 초기화 블록에 추가 (MetricsBuffer 선언 다음):
   fb = FeedbackRenderer(fps=video_fps)
   # 전문가 최저점 y좌표 설정
   from feedback_overlay import extract_expert_depth_target
   target_y_norm = extract_expert_depth_target(ex_frames, EXERCISE)
   # frame_h는 루프 진입 전에 미리 계산하거나 첫 프레임에서 설정

3. 메인 루프 STEP 5 부분 교체 (아래 코드로):
"""

# ── 기존 STEP 5 (교체 전) ──────────────────────────────────────
OLD_CODE = """
# STEP 5a: 전문가 스켈레톤 (좌표 기반 오버레이)
skeleton_visible = check_visibility(lms)

if skeleton_visible:
    draw_expert_skeleton(view, ex_lms_raw, lms, bad_indices, alpha=EXPERT_ALPHA)

# STEP 5b: 사용자 스켈레톤
draw_user_skeleton(view, lms, bad_indices)
frame[:, :target_w] = view
"""

# ── 교체 후 코드 ──────────────────────────────────────────────
NEW_CODE = """
# STEP 5a: 전문가 스켈레톤
skeleton_visible = check_visibility(lms)

if skeleton_visible:
    draw_expert_skeleton(view, ex_lms_raw, lms, bad_indices, alpha=EXPERT_ALPHA)

# STEP 5b: 사용자 스켈레톤 (heatmap OFF일 때만 기본 그리기)
if not fb.is_on('heatmap'):
    draw_user_skeleton(view, lms, bad_indices)

# STEP 5c: 피드백 오버레이 (FeedbackRenderer)
# raw 지표를 넘겨서 내부에서 1euro 스무딩 적용
fb.render(
    view,
    user_metrics_raw = raw,        # compute_metrics 직후 raw 값
    expert_metrics   = ex_metrics,
    deltas           = deltas,
    lms              = lms,
    connections      = PoseLandmarksConns,
    metric_to_lm     = METRIC_TO_LM,
    exercise         = EXERCISE,
)

frame[:, :target_w] = view
"""

# ── 초기화 코드 (루프 직전에 삽입) ──────────────────────────────
INIT_CODE = """
# FeedbackRenderer 초기화
from feedback_overlay import FeedbackRenderer, extract_expert_depth_target
fb = FeedbackRenderer(fps=video_fps)

# 기본 활성화: heatmap + rom
# (필요에 따라 아래 주석 해제)
# fb.enable('depth')
# fb.enable('metricbar')
# fb.disable('rom')

# 깊이선 목표 설정 (전문가 최저점 힙 y좌표)
_target_y_norm = extract_expert_depth_target(ex_frames, EXERCISE)
# frame_h는 첫 프레임 읽은 후 설정:
# fb.depth_line.set_expert_target(_target_y_norm, TARGET_H)
"""

# ── 리셋 키 처리에 추가 ─────────────────────────────────────────
RESET_ADDITION = """
# elif key == ord('r'): 블록에 추가:
fb.reset()
"""

# ── raw 지표 저장 위치 ─────────────────────────────────────────
RAW_SAVE = """
# STEP 2 다음에 raw 저장:
raw = compute_metrics(lms)   # ← 이미 있는 코드
# buf.push(raw) 전에 raw를 별도 변수로 보관
# raw는 1euro 스무딩 전 값 → FeedbackRenderer 내부에서 스무딩
"""

# ── 키보드 단축키 추가 ─────────────────────────────────────────
HOTKEY_CODE = """
# elif 블록에 추가:
elif key == ord('1'):
    fb.toggle_pair('rom', 'heatmap')    # 초보자 모드
elif key == ord('2'):
    fb.toggle_pair('heatmap', 'depth')  # 중급자 모드
elif key == ord('3'):
    # 숙련자 모드: heatmap만
    fb.disable('rom', 'depth', 'metricbar')
    fb.enable('heatmap')
"""


# ── FeedbackRenderer에 toggle_pair 메서드 추가 ────────────────
# feedback_overlay.py의 FeedbackRenderer에 아래 메서드 추가:
TOGGLE_PAIR_METHOD = """
def toggle_pair(self, *layers):
    \"\"\"여러 레이어를 한 번에 토글\"\"\"
    if any(l in self._on for l in layers):
        self.disable(*layers)
    else:
        self.enable(*layers)
"""


# ================================================================
# 적용 시 주의사항
# ================================================================
NOTES = """
1. raw vs user_avg 구분:
   - raw = compute_metrics(lms) 직후 (1euro 미적용)
   - user_avg = buf.get_avg() (이동평균 적용)
   - FeedbackRenderer에는 raw를 넘겨야 내부에서 1euro 적용됨

2. heatmap 레이어 on/off:
   - heatmap이 ON이면 관절 색상을 heatmap이 담당
   - draw_user_skeleton의 빨간/흰색 이진 표시와 충돌하므로
     heatmap이 ON일 때는 draw_user_skeleton 건너뜀

3. 성능:
   - 1euro filter: O(1) per value, 성능 영향 거의 없음
   - DeltaHeatmap.draw_skeleton: O(33 + connections) ≈ O(100)
   - ROMGauge.draw: O(bar_h) ≈ O(200) line 연산
   → 30fps에서 영향 없음

4. 첫 N프레임:
   - 1euro filter는 첫 호출 시 raw값 그대로 반환
   - 2~3프레임 후부터 스무딩 효과 나타남
   - 급격한 초기값 발생 가능하므로 첫 10프레임은
     fb.render 호출 건너뛰는 것도 방법:
     if frame_idx > 10: fb.render(...)

5. 리셋:
   - 세트 시작 시 fb.reset() 호출
   - 1euro filter 초기화, depth line 리셋
"""

if __name__ == '__main__':
    print("이 파일은 패치 가이드입니다.")
    print("feedback_overlay.py와 함께 사용하세요.")
    print(NOTES)
