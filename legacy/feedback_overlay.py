"""
feedback_overlay.py
===================
운동 보조 시스템 - 시각 피드백 레이어 모듈

설계 원칙:
  1. 떨림 방지: 1euro filter (단순 이동평균보다 지연 적음)
  2. 비교 명확성: 각도/비율 기반, 절대좌표 최소화
  3. 인지 부하 최소화: 레이어 독립적, 최대 2~3개 권장
  4. 타이밍: rep 진행 중 시각만, rep 완료 후 텍스트

사용법:
    renderer = FeedbackRenderer(fps=30)
    renderer.enable('rom', 'heatmap')

    # 메인 루프:
    renderer.render(frame, raw_metrics, expert_metrics,
                    deltas, lms, connections, metric_to_lm)

권장 조합:
    초보자 → rom + heatmap
    중급자 → heatmap + depth
    숙련자 → heatmap 단독
"""

import cv2
import numpy as np
import math
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

# ── PIL 한글 지원 ─────────────────────────────────────────────
try:
    from PIL import Image as _PilImg, ImageDraw as _ImageDraw, ImageFont as _PilFont
    _FONTS = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    _fp = next((p for p in _FONTS if os.path.exists(p)), None)
    def _pf(sz): return _PilFont.truetype(_fp, sz) if _fp else _PilFont.load_default()
    _FSS = _pf(13)   # small
    _PIL_OK = True
except Exception:
    _PIL_OK = False

def _put_kr(frame, text, xy, size=13, rgb=(200,200,200)):
    """PIL로 한글 텍스트 렌더링. PIL 없으면 cv2 fallback."""
    if not _PIL_OK:
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (int(rgb[2]),int(rgb[1]),int(rgb[0])), 1, cv2.LINE_AA)
        return
    pil = _PilImg.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fnt = _pf(size)
    _ImageDraw.Draw(pil).text(xy, text, font=fnt, fill=tuple(rgb))
    cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR, frame)


# ================================================================
# [1] 1euro Filter — 떨림 방지 핵심
# ================================================================
# 논문: Casiez et al. (2012) CHI
#
# 핵심: 속도에 따라 스무딩 강도를 동적으로 조절
#   - 정지 시 (준비 자세): 강하게 스무딩 → 깜빡임 제거
#   - 동작 시 (하강/상승): 약하게 스무딩 → 지연 없음
#
# 비교:
#   이동평균 (window=7):  빠른 동작 시 7프레임 지연 발생
#   1euro filter:         정지=안정, 동작=반응 빠름
#
# 실제 효과 (MediaPipe 기준):
#   raw: ±5~8px 진동
#   1euro (min_cutoff=1.0, beta=0.007): ±1~2px

class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.0,
                 beta=0.007, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None
        self._dx = 0.0

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / (1.0 / self.freq))

    def __call__(self, x):
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) * self.freq
        a_d = self._alpha(self.d_cutoff)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff)
        self._x = a * x + (1 - a) * self._x
        return self._x

    def reset(self):
        self._x = None
        self._dx = 0.0


class PoseFilter:
    """관절 좌표/각도 전체에 1euro filter 적용"""
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007, fps=None):
        self._f: Dict[str, OneEuroFilter] = {}
        self._freq = fps if fps is not None else freq
        self._mc = min_cutoff
        self._b = beta

    def _get(self, key):
        if key not in self._f:
            self._f[key] = OneEuroFilter(self._freq, self._mc, self._b)
        return self._f[key]

    def fv(self, key, val):
        """단일 값 필터링"""
        return self._get(key)(val)

    def fp(self, key, x, y):
        """2D 포인트 필터링"""
        return (self._get(key+'_x')(x), self._get(key+'_y')(y))

    def fm(self, metrics: dict) -> dict:
        """지표 딕셔너리 전체 필터링"""
        return {k: self.fv(k, v) for k, v in metrics.items()}

    def reset(self):
        for f in self._f.values():
            f.reset()


# ================================================================
# [2] ROM 게이지 — 가동범위 % 시각화
# ================================================================
# 위치: 영상 왼쪽 세로 바
#
# 표시 내용:
#   [사용자 현재 깊이 %] 채움 바
#   [전문가 현재 위치]   하늘색 가로선 (E 표시)
#   [목표 깊이]          초록 점선 (goal 표시)
#
# 색상:
#   목표 미달: 파란 계열 (차가움 → 더 내려가라)
#   목표 달성: 초록 (OK)
#
# 떨림 대응:
#   - 무릎 각도 → 1euro 스무딩 후 % 변환
#   - % 자체도 3프레임 이동평균 (미세 진동 추가 제거)
#   → 효과: 바가 부드럽게 움직임, 경계에서 깜빡임 없음

class ROMGauge:
    def __init__(self, pf: PoseFilter,
                 angle_min=90.0, angle_max=170.0, target_pct=80.0):
        self._pf = pf
        self._amin = angle_min
        self._amax = angle_max
        self._tgt = target_pct
        self._buf = deque(maxlen=3)

    def _to_pct(self, angle):
        return max(0.0, min(100.0,
            (self._amax - angle) / (self._amax - self._amin) * 100.0))

    def draw(self, frame, user_angle_raw,
             expert_angle=None, expert_min_angle=None,
             x=14, y=70, bar_h=200, bar_w=14):
        H, W = frame.shape[:2]
        s = self._pf.fv('rom', user_angle_raw)
        self._buf.append(self._to_pct(s))
        pct = float(np.mean(self._buf))

        # 배경
        ov = frame.copy()
        cv2.rectangle(ov, (x, y), (x+bar_w, y+bar_h), (25,25,25), -1)

        # 채움
        fill_h = int(bar_h * pct / 100.0)
        for i in range(fill_h):
            r = i / bar_h
            if r < 0.4:   col = (80,200,80)
            elif r < 0.7: col = (0,180,230)
            else:         col = (0,120,255)
            cv2.line(ov, (x, y+bar_h-i), (x+bar_w, y+bar_h-i), col, 1)
        cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x,y), (x+bar_w,y+bar_h), (70,70,70), 1)

        # 목표선
        t_ang = expert_min_angle or (
            self._amax - (self._amax-self._amin)*self._tgt/100.0)
        t_pct = self._to_pct(t_ang)
        t_y = y + int(bar_h*(1-t_pct/100.0))
        for xi in range(x-2, x+bar_w+3, 4):
            cv2.line(frame, (xi,t_y),(min(xi+2,x+bar_w+3),t_y),
                     (80,220,80), 1)
        cv2.putText(frame, 'goal', (x+bar_w+3, t_y+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80,220,80),
                    1, cv2.LINE_AA)

        # 전문가 현재 마커
        if expert_angle is not None:
            ey = y + int(bar_h*(1-self._to_pct(expert_angle)/100.0))
            cv2.line(frame,(x-3,ey),(x+bar_w+3,ey),(0,200,255),1)
            cv2.putText(frame,'E',(x+bar_w+3,ey+4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,200,255),
                        1,cv2.LINE_AA)

        # % 텍스트 (배경 박스 추가)
        color = (80,220,80) if pct>=t_pct else (120,140,255)
        txt = f'{int(pct)}%'
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(frame, (x-2, y-18), (x+tw+2, y-2), (20,20,20), -1)
        cv2.putText(frame, txt, (x, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color,
                    1, cv2.LINE_AA)
        if pct >= t_pct:
            cv2.rectangle(frame, (x-2, y+bar_h+2), (x+22, y+bar_h+18),
                          (20,20,20), -1)
            cv2.putText(frame, 'OK', (x, y+bar_h+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (80,220,80), 1, cv2.LINE_AA)


# ================================================================
# [3] Delta 히트맵 — 관절 색상 인코딩
# ================================================================
# 기존 이진(빨강/흰색) 방식 문제:
#   - 임계값 경계에서 깜빡임 (jitter + threshold 결합)
#   - "얼마나 틀렸는지" 정보 없음
#
# 연속 색상:
#   초록(delta=0) → 노랑(delta=임계/2) → 빨강(delta>=임계)
#
# 떨림 대응:
#   - delta를 1euro로 스무딩 후 색상 계산
#   - 색상이 연속적이라 미세 진동이 있어도 눈에 안 띔
#   - (이진 방식은 진동이 색상 전환으로 바로 보임)

class DeltaHeatmap:
    def __init__(self, pf: PoseFilter):
        self._pf = pf

    @staticmethod
    def _to_color(norm):
        if norm < 0.5:
            t = norm * 2.0
            return (0, 200, int(t*255))   # 초록→노랑
        else:
            t = (norm-0.5)*2.0
            return (0, int((1-t)*200), 255)  # 노랑→빨강

    def draw_skeleton(self, frame, lms, deltas,
                      metric_to_lm, connections,
                      threshold=15.0):
        H, W = frame.shape[:2]

        # 관절번호 → 스무딩 delta 매핑
        jd: Dict[int, float] = {}
        for metric, raw_d in deltas.items():
            s = self._pf.fv(f'hm_{metric}', abs(raw_d))
            norm = min(1.0, s / threshold)
            for idx in metric_to_lm.get(metric, []):
                jd[idx] = max(jd.get(idx, 0.0), norm)

        # 뼈대
        for conn in connections:
            a, b = lms[conn.start], lms[conn.end]
            if a.visibility < 0.3 or b.visibility < 0.3:
                continue
            avg = (jd.get(conn.start,0)+jd.get(conn.end,0))/2
            col = self._to_color(avg)
            cv2.line(frame,
                     (int(a.x*W),int(a.y*H)),
                     (int(b.x*W),int(b.y*H)),
                     col, 2, cv2.LINE_AA)

        # 관절
        for i, lm in enumerate(lms):
            if lm.visibility < 0.3:
                continue
            col = self._to_color(jd.get(i,0.0))
            px, py = int(lm.x*W), int(lm.y*H)
            cv2.circle(frame,(px,py),5,col,-1,cv2.LINE_AA)
            cv2.circle(frame,(px,py),5,(0,0,0),1,cv2.LINE_AA)


# ================================================================
# [4] 깊이선 — 수평 기준선 비교
# ================================================================
# 전문가 최저점 힙 y좌표를 점선으로 표시
# 사용자 현재 힙을 실선으로 표시
#
# 왜 힙 y좌표?:
#   무릎각도는 "숫자", 힙 y는 "선으로 위치 표현" → 더 직관적
#   PT 현장에서 "저 테이프까지 내려와" 방식과 동일한 인지
#
# 떨림 대응:
#   힙 y를 1euro로 스무딩. min_cutoff=0.5로 낮춰서
#   정지 시 더 강하게 스무딩 → 선이 안 흔들림

class DepthLine:
    def __init__(self, pf: PoseFilter):
        self._pf = pf
        self._target_y = None

    def set_expert_target(self, y_norm, frame_h):
        self._target_y = int(y_norm * frame_h)

    def draw(self, frame, user_hip_y_raw):
        H, W = frame.shape[:2]
        s_y = self._pf.fv('dl_hip', user_hip_y_raw)
        uy = int(s_y * H)

        # 사용자 현재 힙 (실선, 연한 회색)
        cv2.line(frame, (int(W*0.08),uy), (int(W*0.42),uy),
                 (150,150,150), 1, cv2.LINE_AA)

        if self._target_y is None:
            return

        ty = self._target_y
        # 전문가 목표 (점선, 하늘색)
        dash = 8
        for xi in range(int(W*0.08), int(W*0.42), dash*2):
            cv2.line(frame,(xi,ty),(min(xi+dash,int(W*0.42)),ty),
                     (0,200,255),1,cv2.LINE_AA)

        # 남은 거리 화살표 (아직 못 내려온 경우)
        if uy < ty:
            mx = int(W*0.06)
            cv2.arrowedLine(frame,(mx,uy+4),(mx,ty-4),
                            (0,180,255),1,cv2.LINE_AA,tipLength=0.25)
        else:
            cv2.putText(frame,'depth OK',
                        (int(W*0.08),ty-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.33,
                        (80,220,80),1,cv2.LINE_AA)

    def reset(self):
        pass


# ================================================================
# [5] 지표 비교 바 — HUD 숫자를 시각화로 보완
# ================================================================
# 기존: "무릎 각도 109.3d (+32.1)"
# 개선: 전문가(파란 바) / 사용자(흰→빨간 바) 나란히
#
# 위치: 기존 HUD 숫자 옆에 보조로 배치
# 역할: HUD 대체가 아닌 보완 — 운동 중 빠른 확인용
#
# 떨림 대응:
#   - 1euro로 스무딩된 값으로 바 길이 계산
#   - 바 길이 자체도 deque(4) 평균 → 미세 진동 제거

class MetricBar:
    def __init__(self, pf: PoseFilter):
        self._pf = pf
        self._bufs: Dict[str, deque] = {}

    def _sb(self, key, val):
        if key not in self._bufs:
            self._bufs[key] = deque(maxlen=4)
        self._bufs[key].append(val)
        return float(np.mean(self._bufs[key]))

    def draw_one(self, frame, x, y, bar_w,
                 u_val, e_val, val_max, label):
        u_r = min(1.0, u_val/val_max)
        e_r = min(1.0, e_val/val_max)
        ub  = int(self._sb(label+'u', bar_w*u_r))
        eb  = int(self._sb(label+'e', bar_w*e_r))
        diff = abs(u_val - e_val)
        dr   = min(1.0, diff/20.0)

        _put_kr(frame, label, (x, y+1), size=13, rgb=(150,150,150))
        bx = x+36
        # 전문가 (파란 계열)
        if eb: cv2.rectangle(frame,(bx,y+2),(bx+eb,y+7),(160,90,0),-1)
        # 사용자 (흰→빨강)
        uc = (int(180*(1-dr)), int(180*(1-dr)), 180)
        if ub: cv2.rectangle(frame,(bx,y+9),(bx+ub,y+14),uc,-1)
        # 차이 수치 (클 때만)
        if diff > 12:
            s = '+' if u_val>e_val else '-'
            cv2.putText(frame, f'{s}{diff:.0f}',
                        (bx+bar_w+3,y+10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.29,
                        (80,110,255),1,cv2.LINE_AA)

    def draw_all(self, frame, u_metrics, e_metrics,
                 keys, labels, x, y, bar_w=100,
                 val_max=180.0, row_h=18):
        H = frame.shape[0]
        for i, k in enumerate(keys):
            ry = y + i*row_h
            if ry + row_h > H-10:
                break
            s_u = self._pf.fv(f'mb_{k}', u_metrics.get(k,0))
            self.draw_one(frame, x, ry, bar_w,
                          s_u, e_metrics.get(k,0),
                          val_max, labels.get(k,k[:4]))


# ================================================================
# [6] FeedbackRenderer — 통합 렌더러
# ================================================================

class FeedbackRenderer:
    def __init__(self, fps=30.0):
        self.pf        = PoseFilter(fps=fps, min_cutoff=1.0, beta=0.007)
        self.rom       = ROMGauge(self.pf)
        self.heatmap   = DeltaHeatmap(self.pf)
        self.depth_line = DepthLine(self.pf)
        self.metricbar = MetricBar(self.pf)
        self._on       = {'heatmap', 'rom'}   # 기본 활성화

    def enable(self, *layers):
        for l in layers: self._on.add(l)

    def disable(self, *layers):
        for l in layers: self._on.discard(l)

    def is_on(self, layer):
        return layer in self._on

    def reset(self):
        self.pf.reset()

    def render(self, frame, user_metrics_raw, expert_metrics,
               deltas, lms, connections, metric_to_lm,
               exercise='squat'):
        if lms is None:
            return frame

        # 렌더링 순서: 배경 → 스켈레톤 → 사이드 UI
        if 'depth' in self._on:
            hy = (lms[23].y + lms[24].y) / 2
            self.depth_line.draw(frame, hy)

        if 'heatmap' in self._on:
            self.heatmap.draw_skeleton(
                frame, lms, deltas, metric_to_lm,
                connections, threshold=15.0)

        if 'rom' in self._on:
            self.rom.draw(
                frame,
                user_metrics_raw.get('knee_angle', 170.0),
                expert_metrics.get('knee_angle', None))

        if 'metricbar' in self._on:
            H, W = frame.shape[:2]
            km = {'squat':   ['knee_angle','hip_angle','spine_lean','foot_width'],
                  'pushup':  ['elbow_angle','body_angle','spine_lean'],
                  'deadlift':['hip_angle','knee_angle','spine_lean']}
            lb = {'knee_angle':'무릎','hip_angle':'힙  ',
                  'spine_lean':'척추','foot_width':'발폭',
                  'elbow_angle':'팔꿈','body_angle':'몸통'}
            self.metricbar.draw_all(
                frame, user_metrics_raw, expert_metrics,
                km.get(exercise, ['knee_angle','hip_angle']),
                lb, x=W-185, y=215, bar_w=105)

        return frame


# ================================================================
# [7] KneeToeGuard — 무릎-발끝 기준선
# ================================================================
# 기획 이유:
#   스쿼트에서 "무릎이 발끝을 넘어가는 것"이 가장 흔한 오류.
#   논문(ScienceDirect 2025)에서 knee-over-toe가 핵심 감지 대상.
#   수직선 하나라 구현이 단순하고 떨림에 강함.
#   사용자는 "저 선을 넘으면 안 된다"는 규칙을 즉각 이해.
#
# 구현 결정:
#   - 발목 x좌표 → 수직 점선 (전신에 걸쳐)
#   - 무릎 x가 선을 넘으면 선 빨간색 + 무릎 위에 경고 원
#   - 1euro로 발목/무릎 x 스무딩 → 선이 흔들리지 않음
#   - 전문가는 다른 색(파란 점선)으로 비교 기준 제공
#
# 평가:
#   ✅ 측면 촬영에서 매우 직관적
#   ✅ 숫자/텍스트 없이 즉각 인지
#   ⚠ 정면 촬영에서는 x좌표 의미 없음 (측면 전용)
#   ⚠ 좌우 다리 평균 쓰면 비대칭 오류 놓칠 수 있음

class KneeToeGuard:
    def __init__(self, pf: PoseFilter):
        self._pf = pf

    def draw(self, frame, lms,
             expert_ankle_x_norm: Optional[float] = None) -> None:
        """
        lms: 사용자 MediaPipe 랜드마크
        expert_ankle_x_norm: 전문가 발목 x (0~1), 없으면 비교선 안 그림
        """
        H, W = frame.shape[:2]

        # 1euro 스무딩 (좌우 각각)
        l_ankle_x = self._pf.fv('ktg_l_ankle_x', lms[27].x)
        r_ankle_x = self._pf.fv('ktg_r_ankle_x', lms[28].x)
        l_knee_x  = self._pf.fv('ktg_l_knee_x',  lms[25].x)
        r_knee_x  = self._pf.fv('ktg_r_knee_x',  lms[26].x)
        l_knee_y  = self._pf.fv('ktg_l_knee_y',  lms[25].y)

        # 더 심한 쪽 선택
        l_diff = (l_knee_x - l_ankle_x) * W
        r_diff = (r_knee_x - r_ankle_x) * W
        if abs(l_diff) >= abs(r_diff):
            ankle_x_px = int(l_ankle_x * W)
            knee_x_px  = int(l_knee_x  * W)
            knee_y_px  = int(l_knee_y  * H)
            diff_px    = l_diff
        else:
            ankle_x_px = int(r_ankle_x * W)
            knee_x_px  = int(r_knee_x  * W)
            knee_y_px  = int(l_knee_y  * H)
            diff_px    = r_diff

        shoulder_w_px = abs(lms[11].x - lms[12].x) * W
        threshold_px  = max(shoulder_w_px * 0.05, 8.0)  # 최소 8px
        is_over = diff_px > threshold_px

        # ── 수직선 (두께 2) ──
        line_color = (0, 60, 200) if is_over else (80, 200, 80)
        dash = 8
        for yi in range(int(H * 0.05), int(H * 0.95), dash * 2):
            cv2.line(frame,
                     (ankle_x_px, yi),
                     (ankle_x_px, min(yi+dash, H)),
                     line_color, 2, cv2.LINE_AA)

        # ── 전문가 기준선 (하늘색, 있을 때만) ──
        if expert_ankle_x_norm is not None:
            ex_ax = int(expert_ankle_x_norm * W)
            for yi in range(int(H*0.2), int(H*0.8), dash*2):
                cv2.line(frame,(ex_ax,yi),
                         (ex_ax, min(yi+dash,H)),
                         (200,180,0),1,cv2.LINE_AA)

        # ── 무릎 경고 (넘었을 때만) ──
        if is_over:
            severity = min(1.0, diff_px / (shoulder_w_px * 0.2 + 1e-6))
            radius = int(10 + severity * 8)
            cv2.circle(frame,(knee_x_px,knee_y_px),
                       radius,(0,0,220),2,cv2.LINE_AA)
            d = radius // 2
            cv2.line(frame,(knee_x_px-d,knee_y_px-d),
                     (knee_x_px+d,knee_y_px+d),(0,0,220),2,cv2.LINE_AA)
            cv2.line(frame,(knee_x_px+d,knee_y_px-d),
                     (knee_x_px-d,knee_y_px+d),(0,0,220),2,cv2.LINE_AA)
            # 텍스트 배경 박스
            txt = f'+{int(diff_px)}px'
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            tx, ty = knee_x_px+8, knee_y_px-8
            cv2.rectangle(frame,(tx-2,ty-th-2),(tx+tw+2,ty+2),(20,20,20),-1)
            cv2.putText(frame,txt,(tx,ty),
                        cv2.FONT_HERSHEY_SIMPLEX,0.36,
                        (80,100,255),1,cv2.LINE_AA)


# ================================================================
# [8] JointArcGauge — 관절 위 원호 게이지
# ================================================================
# 기획 이유:
#   숫자(109.3°)를 운동 중에 읽기 어려움.
#   부채꼴(arc) 면적으로 각도를 표현하면 즉각 인지 가능.
#   전문가(외부 파란 호)와 사용자(내부 흰 채움)를 겹쳐서
#   면적 차이로 오차를 시각화.
#
# 구현 결정:
#   - 무릎 관절 위에 반원 형태 게이지
#   - 내부 채움(흰): 사용자 각도
#   - 외부 테두리 호(파란): 전문가 각도
#   - 차이가 크면 호가 빨간색으로 변함
#   - 각도를 0~180도 → 호의 0~180도로 매핑
#
# 1euro 적용:
#   각도를 스무딩 후 호 그리기
#   → 부채꼴이 부드럽게 변함, 경계에서 깜빡임 없음
#
# 평가:
#   ✅ 숫자 없이 면적으로 직관 표현
#   ✅ 전문가/사용자 차이를 겹쳐서 한눈에
#   ⚠ 관절이 화면 밖이면 안 보임
#   ⚠ 여러 관절에 동시 그리면 복잡 → 무릎 하나만 권장

class JointArcGauge:
    def __init__(self, pf: PoseFilter):
        self._pf = pf

    def draw(self, frame, joint_px: Tuple[int,int],
             user_angle_raw: float,
             expert_angle: float,
             key: str,
             radius: int = 28) -> None:
        """
        joint_px    : (x, y) 픽셀 좌표
        user_angle  : 사용자 각도 (예: 무릎각도 90~170)
        expert_angle: 전문가 각도
        radius      : 게이지 반지름 (px)
        """
        # 1euro 스무딩
        s_user = self._pf.fv(key + '_arc_u', user_angle_raw)
        # 전문가는 이미 스무딩된 값이므로 그대로 사용

        jx, jy = int(joint_px[0]), int(joint_px[1])

        # 각도 → 호 각도 변환 (OpenCV ellipse: 0=오른쪽, 반시계)
        # 무릎각도 90도 → 게이지 90% 채움 (많이 구부림 = 많이 채움)
        # 무릎각도 170도 → 게이지 10% 채움 (거의 안 구부림)
        angle_min, angle_max = 70.0, 175.0

        def to_sweep(angle):
            ratio = 1.0 - (angle - angle_min) / (angle_max - angle_min)
            return max(0.0, min(1.0, ratio)) * 180.0  # 반원

        user_sweep   = to_sweep(s_user)
        expert_sweep = to_sweep(expert_angle)
        diff = abs(s_user - expert_angle)

        # ── 배경 반원 (어두운 회색) ──
        overlay = frame.copy()
        cv2.ellipse(overlay, (jx, jy),
                    (radius, radius), 180, 0, 180,
                    (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # ── 사용자 채움 (흰→빨강) ──
        diff_ratio = min(1.0, diff / 30.0)
        u_color = (
            int(200 * (1 - diff_ratio)),
            int(200 * (1 - diff_ratio)),
            200
        )
        if user_sweep > 1:
            cv2.ellipse(frame, (jx, jy),
                        (radius, radius), 180, 0, int(user_sweep),
                        u_color, -1, cv2.LINE_AA)

        # ── 전문가 외부 호 (파란색) ──
        exp_color = (200, 120, 0) if diff < 15 else (80, 80, 255)
        if expert_sweep > 1:
            cv2.ellipse(frame, (jx, jy),
                        (radius+4, radius+4), 180, 0, int(expert_sweep),
                        exp_color, 2, cv2.LINE_AA)

        # ── 테두리 ──
        cv2.ellipse(frame, (jx, jy),
                    (radius, radius), 180, 0, 180,
                    (80, 80, 80), 1, cv2.LINE_AA)

        # ── 차이 텍스트 (배경 박스 포함) ──
        if diff > 12:
            txt = f'{diff:.0f}deg'
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            tx = jx + radius + 6
            ty = jy + 4
            cv2.rectangle(frame,(tx-2,ty-th-2),(tx+tw+2,ty+2),(20,20,20),-1)
            cv2.putText(frame, txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (80, 120, 255) if diff > 20 else (160, 160, 160),
                        1, cv2.LINE_AA)


# ================================================================
# [9] FormScoreBar — 실시간 자세 점수 바
# ================================================================
# 기획 이유:
#   논문(MDPI 2026)에서 "form accuracy percentage" 단일 지표가
#   216명 실험에서 가장 효과적인 피드백으로 검증됨.
#   여러 지표를 하나의 점수로 압축 → 인지 부하 최소화.
#   사용자는 "지금 몇 점인지" 하나만 보면 됨.
#
# 구현 결정:
#   - 각 지표별 delta/threshold를 정규화
#   - 가중치 합산 → 0~100% 점수
#   - 화면 우상단에 가로 바 + 숫자
#   - 색상: 80% 이상=초록, 60~80%=노랑, 60% 미만=빨강
#   - 1euro로 점수 스무딩 → 바가 안 떨림
#
# 가중치 설계 (스쿼트 기준):
#   무릎각도: 30% (핵심)
#   힙각도:   25% (핵심)
#   척추기울기: 20%
#   발폭:     15%
#   무릎발끝:  10%
#
# 평가:
#   ✅ 단순, 직관적 (게임 점수처럼)
#   ✅ 1euro로 안정적
#   ✅ 논문에서 효과 검증됨
#   ⚠ "왜 점수가 낮은지" 모름 → 다른 레이어와 병행 필요
#   ⚠ 가중치가 종목마다 달라야 함

class FormScoreBar:
    WEIGHTS = {
        'squat':    {'knee_angle':0.30,'hip_angle':0.25,
                     'spine_lean':0.20,'foot_width':0.15,'knee_align':0.10},
        'deadlift': {'hip_angle':0.35,'knee_angle':0.20,
                     'spine_lean':0.30,'foot_width':0.15},
        'pushup':   {'elbow_angle':0.35,'body_angle':0.30,
                     'spine_lean':0.25,'grip_width':0.10},
    }
    THRESHOLDS = {
        'knee_angle':12.0,'hip_angle':12.0,'spine_lean':8.0,
        'foot_width':0.15,'knee_align':0.1,'elbow_angle':12.0,
        'body_angle':10.0,'grip_width':0.15,
    }

    def __init__(self, pf: PoseFilter):
        self._pf  = pf
        self._buf = deque(maxlen=5)  # 추가 평탄화

    def _calc_score(self, deltas: dict, exercise: str) -> float:
        weights = self.WEIGHTS.get(exercise, self.WEIGHTS['squat'])
        total_w, weighted_ok = 0.0, 0.0
        for key, w in weights.items():
            delta = abs(deltas.get(key, 0.0))
            thr   = self.THRESHOLDS.get(key, 15.0)
            # dead zone: 임계값의 20% 이내는 완전 OK로 처리
            dead  = thr * 0.2
            if delta <= dead:
                ok_ratio = 1.0
            else:
                ok_ratio = max(0.0, 1.0 - (delta - dead) / (thr - dead))
            weighted_ok += ok_ratio * w
            total_w += w
        return (weighted_ok / total_w * 100.0) if total_w > 0 else 100.0

    def draw(self, frame, deltas: dict, exercise: str = 'squat',
             x: int = None, y: int = 10, bar_w: int = 120) -> None:
        H, W = frame.shape[:2]
        if x is None:
            x = W - bar_w - 10

        # 점수 계산 + 스무딩
        raw_score = self._calc_score(deltas, exercise)
        s_score   = self._pf.fv('form_score', raw_score)
        self._buf.append(s_score)
        score = float(np.mean(self._buf))

        # 색상
        if score >= 80:
            bar_color = (80, 200, 80)
            txt_color = (80, 220, 80)
        elif score >= 60:
            bar_color = (0, 180, 230)
            txt_color = (0, 200, 255)
        else:
            bar_color = (0, 80, 220)
            txt_color = (0, 100, 255)

        # ── 배경 박스 (텍스트 포함 전체) ──
        cv2.rectangle(frame, (x-42, y-2), (x+bar_w+58, y+16),
                      (20,20,20), -1)
        cv2.rectangle(frame, (x-42, y-2), (x+bar_w+58, y+16),
                      (55,55,55), 1)

        # ── 배경 바 ──
        cv2.rectangle(frame, (x, y+1), (x+bar_w, y+13),
                      (45, 45, 45), -1)

        # ── 채움 바 ──
        fill_w = int(bar_w * score / 100.0)
        if fill_w > 0:
            cv2.rectangle(frame, (x, y+1), (x+fill_w, y+13),
                          bar_color, -1)

        # ── 점수 텍스트 ──
        cv2.putText(frame, f'{int(score)}%',
                    (x + bar_w + 5, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    txt_color, 1, cv2.LINE_AA)

        # ── 레이블 ──
        cv2.putText(frame, 'FORM',
                    (x - 38, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (160, 160, 160), 1, cv2.LINE_AA)


# ================================================================
# [10] FeedbackRenderer 업데이트 (3개 레이어 추가)
# ================================================================

class FeedbackRenderer:
    def __init__(self, fps=30.0):
        self.pf         = PoseFilter(fps=fps, min_cutoff=1.0, beta=0.007)
        self.rom        = ROMGauge(self.pf)
        self.heatmap    = DeltaHeatmap(self.pf)
        self.depth_line = DepthLine(self.pf)
        self.metricbar  = MetricBar(self.pf)
        self.ktg        = KneeToeGuard(self.pf)      # 신규
        self.arc        = JointArcGauge(self.pf)     # 신규
        self.score_bar  = FormScoreBar(self.pf)      # 신규
        self._on        = {'heatmap', 'rom', 'score_bar'}  # 기본 활성화

    def enable(self, *layers):
        for l in layers: self._on.add(l)

    def disable(self, *layers):
        for l in layers: self._on.discard(l)

    def is_on(self, layer):
        return layer in self._on

    def reset(self):
        self.pf.reset()

    def render(self, frame, user_metrics_raw, expert_metrics,
               deltas, lms, connections, metric_to_lm,
               exercise='squat'):
        if lms is None:
            return frame

        H, W = frame.shape[:2]

        # ── 배경 레이어 ──
        if 'depth' in self._on:
            hy = (lms[23].y + lms[24].y) / 2
            self.depth_line.draw(frame, hy)

        if 'ktg' in self._on:
            # 전문가 발목 x (현재 프레임 기준)
            ex_ankle_x = expert_metrics.get('ankle_x_norm', None)
            self.ktg.draw(frame, lms, ex_ankle_x)

        # ── 스켈레톤 레이어 ──
        if 'heatmap' in self._on:
            self.heatmap.draw_skeleton(
                frame, lms, deltas, metric_to_lm,
                connections, threshold=15.0)

        # ── 관절 게이지 ──
        if 'arc' in self._on:
            # 무릎 관절 위치
            lknee_x = self.pf.fv('arc_lkx', lms[25].x) * W
            lknee_y = self.pf.fv('arc_lky', lms[25].y) * H
            self.arc.draw(
                frame,
                joint_px=(lknee_x, lknee_y),
                user_angle_raw=user_metrics_raw.get('knee_angle', 170.0),
                expert_angle=expert_metrics.get('knee_angle', 170.0),
                key='lknee',
                radius=26
            )

        # ── 사이드 UI ──
        if 'rom' in self._on:
            self.rom.draw(
                frame,
                user_metrics_raw.get('knee_angle', 170.0),
                expert_metrics.get('knee_angle', None))

        if 'score_bar' in self._on:
            self.score_bar.draw(frame, deltas, exercise,
                                x=W-130, y=8, bar_w=110)

        if 'metricbar' in self._on:
            km = {'squat':   ['knee_angle','hip_angle','spine_lean','foot_width'],
                  'pushup':  ['elbow_angle','body_angle','spine_lean'],
                  'deadlift':['hip_angle','knee_angle','spine_lean']}
            lb = {'knee_angle':'무릎','hip_angle':'힙  ',
                  'spine_lean':'척추','foot_width':'발폭',
                  'elbow_angle':'팔꿈','body_angle':'몸통'}
            self.metricbar.draw_all(
                frame, user_metrics_raw, expert_metrics,
                km.get(exercise, ['knee_angle','hip_angle']),
                lb, x=W-185, y=215, bar_w=105)

        return frame


# ================================================================
# [12] PhaseIndicator — 동작 단계 표시
# ================================================================
# 기획:
#   스쿼트는 준비→하강→최저점→상승 4단계로 구성됨.
#   현재 단계를 알면 "지금 뭘 해야 하는지"가 명확해지고
#   단계별로 다른 피드백을 줄 수도 있음.
#   무릎각도 + 변화율로 판단: 각도가 작아지면 하강, 커지면 상승.
#
# 구현:
#   1euro 스무딩된 무릎각도의 프레임 간 변화율로 phase 결정.
#   화면 우측에 4개 블록으로 표시. 현재 단계 강조.
#
# 평가:
#   ✅ 기존 knee_angle만으로 판단 가능, 추가 데이터 없음
#   ✅ 단계별 색상으로 직관적
#   ⚠ 각도 변화율이 작을 때 phase 경계에서 깜빡임 가능
#      → dead zone 적용으로 해결

class PhaseIndicator:
    PHASES = ['준비', '하강', '최저점', '상승']
    COLORS = [
        (120, 120, 120),   # 준비: 회색
        (0,   150, 255),   # 하강: 파란색
        (0,   80,  220),   # 최저점: 진한 파란색
        (80,  220, 80),    # 상승: 초록색
    ]
    DOWN_THR = 115.0   # 최저점 판단 무릎각도 임계값
    DEAD_ZONE = 0.003  # 변화율 dead zone (이하면 정지로 판단)

    def __init__(self, pf: PoseFilter):
        self._pf     = pf
        self._prev   = None
        self._phase  = 0   # 0=준비, 1=하강, 2=최저점, 3=상승
        self._stable = 0   # 연속 같은 phase 프레임 수

    def update(self, knee_angle_raw: float) -> int:
        s = self._pf.fv('phase_knee', knee_angle_raw)
        if self._prev is None:
            self._prev = s
            return self._phase

        d = s - self._prev   # 양수=각도 증가=상승, 음수=각도 감소=하강
        self._prev = s

        if s < self.DOWN_THR:
            new_phase = 2  # 최저점
        elif d < -self.DEAD_ZONE:
            new_phase = 1  # 하강 중
        elif d > self.DEAD_ZONE:
            new_phase = 3  # 상승 중
        else:
            new_phase = 0  # 준비/정지

        if new_phase == self._phase:
            self._stable = min(self._stable + 1, 10)
        else:
            self._stable -= 2
            if self._stable <= 0:
                self._stable = 0
                self._phase  = new_phase

        return self._phase

    def draw(self, frame, phase: int,
             x: int = None, y: int = 30,
             block_h: int = 18, block_w: int = 52) -> None:
        H, W = frame.shape[:2]
        if x is None:
            x = W - block_w - 8

        for i, (name, col) in enumerate(zip(self.PHASES, self.COLORS)):
            ry = y + i * (block_h + 3)
            active = (i == phase)

            # 배경
            bg = col if active else (25, 25, 25)
            cv2.rectangle(frame, (x, ry), (x+block_w, ry+block_h), bg, -1)
            cv2.rectangle(frame, (x, ry), (x+block_w, ry+block_h),
                          col if active else (60,60,60), 1)

            # 텍스트
            txt_col = (255,255,255) if active else (100,100,100)
            _put_kr(frame, name, (x+4, ry+2), size=12, rgb=txt_col)

            # 활성 인디케이터 (왼쪽 바)
            if active:
                cv2.rectangle(frame, (x-4, ry), (x-2, ry+block_h), col, -1)


# ================================================================
# [13] SymmetryBar — 좌우 대칭 바
# ================================================================
# 기획:
#   좌/우 무릎 y좌표 차이로 골반 수평 여부 판단.
#   한쪽으로 치우쳐 앉는 오류를 즉각 감지.
#   중앙 정렬 바로 직관적 표시.
#
# 구현:
#   lms[25].y - lms[26].y (왼무릎 - 오른무릎) 차이를 1euro 스무딩.
#   차이가 클수록 바가 한쪽으로 치우침.
#
# 평가:
#   ✅ 추가 계산 없이 기존 lms만으로 구현
#   ✅ 직관적 (가운데가 좋음을 직관적으로 인지)
#   ⚠ 측면 촬영에서 좌우 무릎 y 차이는 카메라 각도 영향받음
#      → 허용 범위 넓게 설정 (dead zone)

class SymmetryBar:
    def __init__(self, pf: PoseFilter):
        self._pf = pf

    def draw(self, frame, lms,
             x: int = None, y: int = None,
             bar_w: int = 100) -> None:
        H, W = frame.shape[:2]
        if x is None: x = W//2 - bar_w//2
        if y is None: y = H - 30

        l_y = self._pf.fv('sym_ly', lms[25].y)
        r_y = self._pf.fv('sym_ry', lms[26].y)
        diff = l_y - r_y   # 양수 = 왼쪽이 더 낮음

        # 정규화 (-0.05~0.05 → -1~1)
        ratio = max(-1.0, min(1.0, diff / 0.05))
        is_ok = abs(ratio) < 0.3

        # ── 배경 바 ──
        cv2.rectangle(frame, (x, y), (x+bar_w, y+10), (30,30,30), -1)
        cv2.rectangle(frame, (x, y), (x+bar_w, y+10), (60,60,60), 1)

        # ── 채움 (중앙 기준) ──
        cx = x + bar_w//2
        fill = int(abs(ratio) * bar_w//2)
        col = (80,220,80) if is_ok else (0,80,220)
        if ratio > 0:
            cv2.rectangle(frame, (cx, y+1), (cx+fill, y+9), col, -1)
        else:
            cv2.rectangle(frame, (cx-fill, y+1), (cx, y+9), col, -1)

        # 중앙선
        cv2.line(frame, (cx, y-2), (cx, y+12), (180,180,180), 1)

        # 레이블
        _put_kr(frame, 'L', (x-14, y), size=11, rgb=(160,160,160))
        _put_kr(frame, 'R', (x+bar_w+3, y), size=11, rgb=(160,160,160))
        if not is_ok:
            side = 'L' if ratio > 0 else 'R'
            cv2.putText(frame, f'{side} tilt', (x, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (0,100,255), 1, cv2.LINE_AA)


# ================================================================
# [14] TempoWave — 각도 변화율 파형
# ================================================================
# 기획:
#   반동(너무 빠른 하강)이나 잠금(너무 느린 상승)을 감지.
#   각도 변화율을 파형으로 표시해서 리듬을 시각화.
#   적정 속도 범위를 점선으로 표시.
#
# 구현:
#   무릎각도 변화율 (d_angle/frame)을 deque(40)에 저장.
#   화면 상단에 작은 파형 그래프로 표시.
#   ±임계값 초과 시 빨간색.
#
# 평가:
#   ✅ 기존 1euro 필터 출력값 사용, 추가 계산 없음
#   ✅ 반동 감지에 효과적
#   ⚠ 적정 속도 임계값이 종목/개인마다 다름 → 상대값 사용

class TempoWave:
    def __init__(self, pf: PoseFilter, buf_size: int = 40):
        self._pf   = pf
        self._prev = None
        self._buf  = deque(maxlen=buf_size)
        self.SPEED_THR = 3.0   # 도/프레임, 이 이상이면 빠름

    def update(self, knee_angle_raw: float) -> float:
        s = self._pf.fv('tempo_knee', knee_angle_raw)
        d = 0.0
        if self._prev is not None:
            d = s - self._prev
        self._prev = s
        self._buf.append(d)
        return d

    def draw(self, frame,
             x: int = None, y: int = 28,
             w: int = 120, h: int = 28) -> None:
        H, W = frame.shape[:2]
        if x is None: x = W//2 - w//2
        n = len(self._buf)
        if n < 2:
            return

        # 배경
        ov = frame.copy()
        cv2.rectangle(ov, (x, y), (x+w, y+h), (20,20,20), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,50), 1)

        # 중앙선 (0 속도)
        mid_y = y + h//2
        cv2.line(frame, (x, mid_y), (x+w, mid_y), (70,70,70), 1)

        # 임계값 점선
        thr_px = int(h/2 * self.SPEED_THR / 5.0)
        for xi in range(x, x+w, 6):
            cv2.line(frame, (xi, mid_y-thr_px), (xi+3, mid_y-thr_px),
                     (0,80,150), 1)
            cv2.line(frame, (xi, mid_y+thr_px), (xi+3, mid_y+thr_px),
                     (0,80,150), 1)

        # 파형
        pts = list(self._buf)
        max_val = max(abs(v) for v in pts) if pts else self.SPEED_THR
        max_val = max(max_val, self.SPEED_THR * 0.5)  # 최소 스케일
        scale = (h//2 - 2) / max_val
        for i in range(1, n):
            px0 = x + int((i-1) / n * w)
            px1 = x + int(i     / n * w)
            py0 = mid_y - int(pts[i-1] * scale)
            py1 = mid_y - int(pts[i]   * scale)
            py0 = max(y+1, min(y+h-1, py0))
            py1 = max(y+1, min(y+h-1, py1))
            fast = abs(pts[i]) > self.SPEED_THR
            col  = (0, 60, 220) if fast else (0, 180, 220)
            cv2.line(frame, (px0,py0), (px1,py1), col, 1, cv2.LINE_AA)

        # 레이블
        cv2.putText(frame, 'TEMPO', (x+2, y+h-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                    (100,100,100), 1, cv2.LINE_AA)


# ================================================================
# [15] ACLRiskScore — ACL 부상 위험도 점수
# ================================================================
# 기획:
#   무릎각도(너무 얕음) + 무릎-발끝 정렬(무릎 전방) +
#   척추 기울기(과도한 앞쏠림)를 결합해서 단일 위험도 점수로 표시.
#   논문(PMC 2020): 이 3가지 조합이 ACL 위험 예측에 유효.
#
# 구현:
#   각 요소를 0~1로 정규화 후 가중 평균.
#   낮을수록 안전 (초록), 높을수록 위험 (빨강).
#
# 평가:
#   ✅ 기존 deltas/metrics만으로 계산 가능
#   ✅ 단일 지표라 인지 부하 없음
#   ⚠ 2D 카메라로는 진정한 valgus 측정 불가,
#     knee_align (x 정렬) 로 근사

class ACLRiskScore:
    def __init__(self, pf: PoseFilter):
        self._pf  = pf
        self._buf = deque(maxlen=5)

    def _calc(self, user_metrics: dict, deltas: dict) -> float:
        # 요소 1: 무릎각도 (너무 얕으면 위험, <120도에서 높아짐)
        knee = user_metrics.get('knee_angle', 170.0)
        r_knee = max(0.0, min(1.0, (130 - knee) / 40.0)) if knee < 130 else 0.0

        # 요소 2: 무릎-발끝 정렬 (knee_align delta)
        ka = abs(deltas.get('knee_align', 0.0))
        r_align = min(1.0, ka / 0.15)

        # 요소 3: 척추 기울기 (spine_lean)
        sl = abs(user_metrics.get('spine_lean', 0.0))
        r_spine = min(1.0, sl / 20.0)

        # 가중 평균 (무릎정렬 50%, 무릎각도 30%, 척추 20%)
        return r_align*0.5 + r_knee*0.3 + r_spine*0.2

    def draw(self, frame, user_metrics: dict, deltas: dict,
             x: int = None, y: int = None) -> None:
        H, W = frame.shape[:2]
        if x is None: x = 14
        if y is None: y = H - 90

        raw = self._calc(user_metrics, deltas)
        s   = self._pf.fv('acl_risk', raw)
        self._buf.append(s)
        risk = float(sum(self._buf)/len(self._buf))

        # 색상: 초록→노랑→빨강
        if risk < 0.4:
            col = (80, 200, 80)
            label = 'SAFE'
        elif risk < 0.7:
            col = (0, 180, 220)
            label = 'CAUTION'
        else:
            col = (0, 60, 220)
            label = 'RISK'

        # 배경
        cv2.rectangle(frame, (x-2, y-2), (x+60, y+28),
                      (20,20,20), -1)
        cv2.rectangle(frame, (x-2, y-2), (x+60, y+28),
                      col, 1)

        # ACL 텍스트
        cv2.putText(frame, 'ACL',
                    (x+2, y+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (140,140,140), 1, cv2.LINE_AA)
        cv2.putText(frame, label,
                    (x+2, y+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    col, 1, cv2.LINE_AA)

        # 위험도 바 (세로)
        bar_h = 24
        fill_h = int(bar_h * risk)
        cv2.rectangle(frame, (x+52, y), (x+58, y+bar_h),
                      (40,40,40), -1)
        if fill_h > 0:
            cv2.rectangle(frame, (x+52, y+bar_h-fill_h),
                          (x+58, y+bar_h), col, -1)


# ================================================================
# FeedbackRenderer 업데이트
# ================================================================

class FeedbackRenderer:
    def __init__(self, fps=30.0):
        self.pf         = PoseFilter(fps=fps, min_cutoff=1.0, beta=0.007)
        self.rom        = ROMGauge(self.pf)
        self.heatmap    = DeltaHeatmap(self.pf)
        self.depth_line = DepthLine(self.pf)
        self.metricbar  = MetricBar(self.pf)
        self.ktg        = KneeToeGuard(self.pf)
        self.arc        = JointArcGauge(self.pf)
        self.score_bar  = FormScoreBar(self.pf)
        self.phase      = PhaseIndicator(self.pf)   # 신규
        self.symmetry   = SymmetryBar(self.pf)      # 신규
        self.tempo      = TempoWave(self.pf)        # 신규
        self.acl        = ACLRiskScore(self.pf)     # 신규
        self._on        = {'heatmap', 'rom', 'score_bar'}
        self._cur_phase = 0

    def enable(self, *layers):
        for l in layers: self._on.add(l)

    def disable(self, *layers):
        for l in layers: self._on.discard(l)

    def is_on(self, layer):
        return layer in self._on

    def get_phase(self) -> int:
        return self._cur_phase

    def reset(self):
        self.pf.reset()
        self.tempo._prev = None
        self.phase._prev = None

    def render(self, frame, user_metrics_raw, expert_metrics,
               deltas, lms, connections, metric_to_lm,
               exercise='squat'):
        if lms is None:
            return frame

        H, W = frame.shape[:2]

        # ── Phase 업데이트 (항상) ──
        knee_raw = user_metrics_raw.get('knee_angle', 170.0)
        self._cur_phase = self.phase.update(knee_raw)
        self.tempo.update(knee_raw)

        # ── 배경 레이어 ──
        if 'depth' in self._on:
            hy = (lms[23].y + lms[24].y) / 2
            self.depth_line.draw(frame, hy)

        if 'ktg' in self._on:
            self.ktg.draw(frame, lms)

        # ── 스켈레톤 레이어 ──
        if 'heatmap' in self._on:
            self.heatmap.draw_skeleton(
                frame, lms, deltas, metric_to_lm,
                connections, threshold=15.0)

        # ── 관절 게이지 ──
        if 'arc' in self._on:
            lknee_x = self.pf.fv('arc_lkx', lms[25].x) * W
            lknee_y = self.pf.fv('arc_lky', lms[25].y) * H
            self.arc.draw(
                frame,
                joint_px=(lknee_x, lknee_y),
                user_angle_raw=knee_raw,
                expert_angle=expert_metrics.get('knee_angle', 170.0),
                key='lknee', radius=26)

        # ── 사이드 UI ──
        if 'rom' in self._on:
            self.rom.draw(frame, knee_raw,
                          expert_metrics.get('knee_angle', None))

        if 'score_bar' in self._on:
            self.score_bar.draw(frame, deltas, exercise,
                                x=W-130, y=8, bar_w=110)

        if 'phase' in self._on:
            self.phase.draw(frame, self._cur_phase,
                            x=W-68, y=30)

        if 'symmetry' in self._on:
            self.symmetry.draw(frame, lms,
                               x=8, y=290, bar_w=90)

        if 'tempo' in self._on:
            self.tempo.draw(frame, x=W//2-60, y=10, w=120, h=28)

        if 'acl' in self._on:
            self.acl.draw(frame, user_metrics_raw, deltas,
                          x=14, y=H-90)

        if 'metricbar' in self._on:
            km = {'squat':   ['knee_angle','hip_angle','spine_lean','foot_width'],
                  'pushup':  ['elbow_angle','body_angle','spine_lean'],
                  'deadlift':['hip_angle','knee_angle','spine_lean']}
            lb = {'knee_angle':'무릎','hip_angle':'힙  ',
                  'spine_lean':'척추','foot_width':'발폭',
                  'elbow_angle':'팔꿈','body_angle':'몸통'}
            self.metricbar.draw_all(
                frame, user_metrics_raw, expert_metrics,
                km.get(exercise, ['knee_angle','hip_angle']),
                lb, x=W-185, y=215, bar_w=105)

        return frame
# ================================================================

def extract_expert_depth_target(ex_frames, exercise='squat'):
    """
    전문가 JSON에서 최저점 프레임의 힙 y좌표(0~1) 추출.
    DepthLine.set_expert_target()에 전달하여 사용.
    """
    if not ex_frames:
        return 0.7
    ak = {'squat':'knee_angle','deadlift':'hip_angle',
          'pushup':'elbow_angle'}.get(exercise,'knee_angle')
    best_frame, best_angle = None, float('inf')
    for fd in ex_frames:
        a = fd.get('metrics',{}).get(ak, 180.0)
        if a < best_angle:
            best_angle, best_frame = a, fd
    if best_frame is None:
        return 0.7
    lms = best_frame.get('landmarks', [])
    if len(lms) >= 25:
        return (lms[23][1] + lms[24][1]) / 2.0
    return 0.7