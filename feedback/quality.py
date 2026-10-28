"""Image-space support of the joints used by the existing 2D metric formulas.

Metric values and v4 delta thresholds stay in unified_feedback_v4. This module
only determines whether an observation may contribute to a position median.
"""
from collections import Counter

import numpy as np

from unified_feedback_v4 import SIDE_IDS

VISIBILITY_MIN = .5
TORSO = [11, 12, 23, 24]
JOINT_NAMES = {0: '머리', 11: '어깨', 12: '어깨', 13: '팔꿈치', 14: '팔꿈치',
               15: '손목', 16: '손목', 19: '손가락', 20: '손가락', 21: '손가락', 22: '손가락',
               23: '엉덩이', 24: '엉덩이', 25: '무릎', 26: '무릎', 27: '발목', 28: '발목',
               29: '뒤꿈치', 30: '뒤꿈치', 31: '발끝', 32: '발끝'}


def required_joints(metric, side, *, has_fingers=True):
    ids = SIDE_IDS[side]
    dependencies = {
        'knee_angle': [ids['hip'], ids['knee'], ids['ankle']],
        'hip_angle': [ids['shoulder'], ids['hip'], ids['knee']],
        'elbow_angle': [ids['shoulder'], ids['elbow'], ids['wrist']],
        'lockout_angle_min': [ids['shoulder'], ids['elbow'], ids['wrist']],
        'trunk_lean': TORSO,
        'foot_flatness': [ids['heel'], ids['foot']] + TORSO,
        'head_hip_line': [0] + TORSO,
        'wrist_elbow_x_diff': [ids['elbow'], ids['wrist']] + TORSO,
        'bench_line_diff': [0] + TORSO,
        # The proxy combines both grips, each built from wrist/index/thumb.
        'bar_proxy_conf': [15, 16, 19, 20, 21, 22],
    }
    if metric == 'bar_proxy_conf' and not has_fingers:
        return [15, 16]
    return dependencies[metric]


def observation_reason(lms, metric, side, *, has_fingers=True):
    for joint in required_joints(metric, side, has_fingers=has_fingers):
        point = lms[joint]
        name = JOINT_NAMES[joint]
        if not np.isfinite(point[[0, 1, 3]]).all():
            return f'{name} 관절 관측 없음'
        if not (0 <= point[0] <= 1 and 0 <= point[1] <= 1):
            return f'{name} 관절 화면 밖'
        if point[3] < VISIBILITY_MIN:
            return f'{name} 관절 visibility 부족'
    return None


def window_reason(reasons, valid_count, minimum=3):
    if valid_count >= minimum:
        return None
    failures = [reason for reason in reasons if reason is not None]
    reason = Counter(failures).most_common(1)[0][0] if failures else '관절 유효 프레임 부족'
    return f'{reason} (유효 {valid_count}/5프레임)'
