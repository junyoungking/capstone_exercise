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
from collections import deque
from typing import Dict, List, Optional, Tuple


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

        # % 텍스트
        color = (80,220,80) if pct>=t_pct else (120,140,255)
        cv2.putText(frame, f'{int(pct)}%', (x,y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color,
                    1, cv2.LINE_AA)
        if pct >= t_pct:
            cv2.putText(frame, 'OK', (x, y+bar_h+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
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

        cv2.putText(frame, label, (x, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (130,130,130), 1, cv2.LINE_AA)
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
# [7] 유틸: 전문가 최저점 추출
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