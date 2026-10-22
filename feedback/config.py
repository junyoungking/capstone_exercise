"""설계의 상수와 문구. 기존 지표 설정은 원본 모듈에서 그대로 가져온다."""

from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("data/demo.json")
DEFAULT_OUT = Path("out/demo")
STAGES = ("segments", "metrics", "findings", "report")
DEFAULT_UNTIL = "report"
SEGMENT_MODES = ("labels", "model")
DEFAULT_REF_WIDTH = 300
DEFAULT_REF_QUALITY = 72

# CLI 도움말에는 영상 처리 패키지가 필요 없다. 실제로 설정을 요청할 때만
# 원본 모듈을 import하며, 설정값을 복사하거나 새로 정의하지 않는다.
_REUSED_CONSTANT_SOURCES = {
    "DISPLAY_METRICS": "unified_feedback_v4",
    "DELTA_THRESHOLDS": "unified_feedback_v4",
    "METRIC_LABELS_KO": "unified_feedback_v4",
    "ADVISORY_METRICS": "unified_feedback_v4",
    "RATIO_METRICS": "offline_compare",
    "MIN_VALID_RATIO": "offline_compare",
    "MIN_REP_METRIC_RATIO": "offline_compare",
}


def __getattr__(name: str):
    source = _REUSED_CONSTANT_SOURCES.get(name)
    if source is None:
        raise AttributeError(name)
    value = getattr(import_module(source), name)
    globals()[name] = value
    return value


# 5.1 포지션 및 5.2 측정
POSITIONS = ("START", "MID_DOWN", "BOTTOM", "MID_UP", "LOCKOUT")
POSITION_LABELS = {
    "START": "시작",
    "MID_DOWN": "하강 중간",
    "BOTTOM": "바닥",
    "MID_UP": "상승 중간",
    "LOCKOUT": "락아웃",
}
PHASES = ("down", "bottom", "up")
PHASE_LABELS = {"down": "하강", "bottom": "바닥", "up": "상승"}
HIP_MAX_GAP_FRAMES = 5
HIP_SMOOTH_WINDOW = 5
FACING_SAMPLE_FRAMES = 5
FRAME_PADDING = 15
POSITION_WINDOW_RADIUS = 2
MIN_POSITION_VALUES = 3

MODEL_CONTEXT_FRAMES = 10
MODEL_ENDPOINT_RATIO = 0.05
MODEL_STOP_SPEED_RATIOS = (0.005, 0.01, 0.02)
MODEL_TOP_BAND_RATIO = 0.2
MODEL_STOP_FRAMES = 3
MODEL_SEARCH_EXTRA_FRAMES = 30
MODEL_ENDPOINT_METHODS = {"velocity": "velocity", "fallback": "fallback_5pct", "override": "override"}
MODEL_BOTTOM_RATIO = 0.03
MODEL_MIN_AMPLITUDE_RATIO = 0.3
MODEL_CLIP_LENGTH = 16
MODEL_PHASE_FILTER_WINDOW = 5
MODEL_MIN_PHASE_FRAMES = 5
MODEL_INFERENCE_BATCH_SIZE = 32
MODEL_INFERENCE_THREADS = 4

POSE3D_UI_LABELS = {"robot": "로봇", "stick": "스틱", "side": "측면", "front": "정면",
                    "top": "위", "back": "뒤", "no_data": "3D 데이터 없음", "pip": "원본 영상"}
POSE3D_METRIC_JOINTS = {
    "knee_angle": {"right": [1, 2, 3], "left": [4, 5, 6]},
    "hip_angle": {"right": [14, 1, 2], "left": [11, 4, 5]},
    "trunk_lean": {"right": [0, 7, 8], "left": [0, 7, 8]},
    "foot_flatness": {"right": [3], "left": [6]},
    "head_hip_line": {"right": [10, 0], "left": [10, 0]},
}
POSE3D_JOINTS = (
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)
POSE3D_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
POSE3D_MAX_INTERPOLATE_GAP = 10
POSE3D_RTMLIB_MODE = "lightweight"
POSE3D_MODEL_FILES = {
    "lightweight": {
        "det": "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx",
        "pose": "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx",
        "local_det": "yolox_tiny.onnx",
        "local_pose": "rtmpose_s.onnx",
    },
    "balanced": {"local_det": "yolox_m.onnx", "local_pose": "rtmpose_m.onnx"},
}
POSE3D_DET_INPUT_SIZES = {"lightweight": (416, 416), "balanced": (640, 640)}
POSE3D_POSE_INPUT_SIZES = {"lightweight": (192, 256), "balanced": (192, 256)}
POSE3D_THREADS = 4
POSE3D_CHECKPOINT_NAME = "pretrained_h36m_detectron_coco.bin"
POSE3D_VENDOR_MODEL = PROJECT_ROOT / "track3d" / "vendor" / "videopose3d" / "model.py"
POSE3D_VIDEPOSE_FILTER_WIDTHS = (3, 3, 3, 3, 3)
POSE3D_VIDEPOSE_CHANNELS = 1024

# 5.3 비교 및 5.4 선택
K_NONE_MAX = 1.0
K_YELLOW_MAX = 1.5
TEMPO_THRESHOLDS = {"down_s": 0.3, "bottom_s": 0.2, "up_s": 0.3}
GROUPS = {"depth": {"where": "BOTTOM", "metrics": ("knee_angle", "hip_angle")}}
MAX_EVIDENCE = 3
RATIO_PERCENT_SCALE = 100

# 5.5 서술: 문장 생성은 이후 describe.py에서 이 템플릿을 사용한다.
METRIC_TEMPLATES = {
    "knee_angle": {
        "+": "무릎을 레퍼런스보다 {n}° 덜 굽혔습니다",
        "-": "무릎을 레퍼런스보다 {n}° 더 굽혔습니다",
    },
    "hip_angle": {
        "+": "엉덩이를 레퍼런스보다 {n}° 덜 접었습니다",
        "-": "엉덩이를 레퍼런스보다 {n}° 더 접었습니다",
    },
    "trunk_lean": {
        "+": "상체를 레퍼런스보다 {n}° 더 숙였습니다",
        "-": "상체를 레퍼런스보다 {n}° 덜 숙였습니다",
    },
    "foot_flatness": {
        "+": "발 앞뒤 높이 차이가 레퍼런스보다 몸통 길이의 {n}%만큼 큽니다",
        "-": "발 앞뒤 높이 차이가 레퍼런스보다 몸통 길이의 {n}%만큼 작습니다",
    },
    "head_hip_line": {
        "+": "머리와 엉덩이의 가로 거리가 레퍼런스보다 몸통 길이의 {n}%만큼 멉니다",
        "-": "머리와 엉덩이의 가로 거리가 레퍼런스보다 몸통 길이의 {n}%만큼 가깝습니다",
    },
}
SINGLE_TEMPLATE = "{where}에서 {phrase}."
DEPTH_TEMPLATES = {
    "+": "바닥에서 레퍼런스보다 덜 깊게 앉았습니다 (무릎 {knee}°, 엉덩이 {hip}°).",
    "-": "바닥에서 레퍼런스보다 더 깊게 앉았습니다 (무릎 {knee}°, 엉덩이 {hip}°).",
}
NONE_SENTENCE = "모든 포지션에서 레퍼런스와의 차이가 임계값 안에 있습니다."
TEMPO_MOVEMENT_TEMPLATE = (
    "{phase}이 레퍼런스보다 {n:.1f}초 {direction} "
    "(하강:상승 {user_ratio:.1f}:1, 레퍼런스 {ref_ratio:.1f}:1)."
)
TEMPO_MOVEMENT_DIRECTIONS = {"+": "느립니다", "-": "빠릅니다"}
TEMPO_BOTTOM_TEMPLATE = "바닥에서 레퍼런스보다 {n:.1f}초 {direction} 머물렀습니다."
TEMPO_BOTTOM_DIRECTIONS = {"+": "더", "-": "덜"}
EXCLUDE_TEMPLATE = "이 rep은 비교에서 뺐습니다 — {reason}."
DETECTION_REASON_TEMPLATE = "{phase} 구간 관절 검출 {ratio:.0%} (기준 {minimum:.0%})"
METRIC_REASON_TEMPLATE = "측정 가능한 칸 {ratio:.0%} (기준 {minimum:.0%})"
REF_EXCLUDE_REASON = "레퍼런스 rep 검출 부족"
TAB_SINGLE_TEMPLATE = "{where} · {metric} {sign}{n}{unit}"
TAB_DEPTH_LABELS = {"+": "바닥 · 덜 깊게", "-": "바닥 · 더 깊게"}
TAB_NONE_LABEL = "차이 없음"
TAB_EXCLUDED_LABEL = "제외"
BANNED = (
    "좋", "나쁘", "잘못", "틀렸", "틀린", "올바", "정확한 자세",
    "해야", "하세요", "하십시오", "권장", "추천",
)

# CLI 문구
CLI_DESCRIPTION = "사용자 rep과 레퍼런스 1 rep의 세트 후 비교 리포트"
CLI_HELP = {
    "config": "프로젝트 루트 기준 설정 JSON 경로",
    "out": "프로젝트 루트 기준 출력 폴더",
    "until": "마지막 처리 단계 (기본: report)",
    "segment": "분할 모드 (생략하면 양쪽 reps 유무로 선택)",
    "override": "포지션 수정 JSON 경로",
    "ref_width": "레퍼런스 스프라이트 가로 크기 (기본: 300)",
    "ref_quality": "레퍼런스 JPEG 품질 (기본: 72)",
    "no_cache": "포즈 캐시를 무시하고 다시 추출",
    "ckpt": "model 모드에서 사용하는 ST-GCN 체크포인트 경로",
}
CONFIG_ERROR_TEMPLATE = "설정 파일을 읽을 수 없습니다: {path} — {reason}"
MODEL_CHECKPOINT_REQUIRED = "모델 모드에는 --ckpt로 ST-GCN 체크포인트 경로를 지정합니다."
MODEL_UNAVAILABLE_TEMPLATE = "ST-GCN 추론을 시작할 수 없습니다: {reason}"

# 7단계 입력 검사·실패 화면
VIEW_SIDE_MAX = 0.35
VIEW_FRONT_MIN = 0.60
VIDEO_FPS_TOLERANCE = 0.2
VIDEO_FRAME_COUNT_TOLERANCE = 1
VIDEO_TEN_BIT_MARKERS = ("10le", "10be")
VIDEO_TEN_BIT_PROFILES = ("High 10",)
ROLE_LABELS = {"user": "사용자", "ref": "레퍼런스"}
VIEW_LABELS = {"side": "측면", "diagonal": "대각", "front": "정면"}
FAILURE_TEMPLATE_PATH = PROJECT_ROOT / "web" / "failure_template.html"
FAILURE_TITLE = "분석할 수 없습니다"
FAILURE_CHECK_LABEL = "입력 검사"
FAILURE_CHECKLIST = (
    "측면에서 90°로 촬영했는지",
    "전신과 원판이 다 보이는지",
    "너무 멀지 않은지",
    "뒤에 다른 사람이 없는지",
    "고정 fps·H.264인지 (tools/check_video.py --fix는 승인 후 실행)",
)
GATE_REMEDIES = {
    1: "영상 경로와 tools/check_video.py의 진단 결과를 확인한 뒤 다시 실행.",
    2: "같은 종목의 사용자 영상과 레퍼런스 영상으로 다시 실행.",
    3: "두 영상의 촬영 각도와 설정의 view 값을 맞춘 뒤 다시 실행.",
    4: "사람의 전신이 보이는 영상으로 다시 실행.",
    5: "관절이 보이는 rep와 레퍼런스 rep을 선택한 뒤 다시 실행.",
}
FAILURE_DEFAULT_REMEDY = "영상 경로와 설정의 프레임 범위를 확인한 뒤 다시 실행."
GATE_TEXT = {
    "no_view": "없음",
    "video_open": "{role} 영상을 열 수 없습니다: {video} ({reason}).",
    "video_empty": "{role} 영상에서 읽은 프레임이 {actual}개입니다 (기준 1개 이상).",
    "probe_failed": "{role} 영상 정보를 읽을 수 없습니다: {reason}.",
    "probe_missing": "{role}: ffprobe가 없어 코덱·10bit·가변 fps·회전 검사를 건너뛰었습니다.",
    "codec": "{role} 영상 코덱({codec})은 브라우저에서 지원되지 않습니다.",
    "codec_warning": "{role}: 영상 코덱({codec})의 브라우저 지원을 확인하지 못했습니다.",
    "ten_bit": "{role} 영상은 10bit입니다 ({pix_fmt}, {profile}).",
    "variable_fps": "{role} 영상의 fps가 일정하지 않습니다 (표기 {rate:.3f}, 평균 {average:.3f}, 차이 기준 {maximum:.1f} fps).",
    "rotation": "{role} 영상에 회전 정보({rotation}°)가 있습니다.",
    "frame_count": "{role} 영상의 표기 프레임 수({count})와 읽은 수({actual})가 다릅니다 (차이 기준 {maximum}프레임).",
    "size": "{role} 영상 크기가 다릅니다 (ffprobe {probe_w}×{probe_h}, OpenCV {cv_w}×{cv_h}).",
    "exercise": "두 영상의 종목이 다릅니다 (사용자 {user}, 레퍼런스 {ref}).",
    "view": "두 영상의 촬영 각도 설정이 다릅니다 (사용자 {user}, 레퍼런스 {ref}).",
    "unknown_view": "{role}의 촬영 각도 설정({view})을 읽을 수 없습니다.",
    "detection": "{role} 영상의 관절 검출률이 {detected}/{total}프레임({ratio:.1%})입니다 (기준 {minimum:.0%} 이상).",
    "ref_missing": "선택한 레퍼런스 rep {rep}을 찾을 수 없습니다 (일치 {matches}개, 전체 {total}개).",
    "ref_excluded": "선택한 레퍼런스 rep {rep}은 비교에서 제외됐습니다: {reason}.",
    "user_excluded": "비교 가능한 사용자 rep이 {comparable}/{total}개입니다 (기준 1개 이상).",
    "angle_warning": "{role}: 어깨 너비/몸통 길이 {ratio:.3f}로 {actual}으로 분류됐습니다 (설정 {configured}).",
    "angle_missing": "{role}: START의 유효 관절이 없어 촬영 각도를 자동 확인하지 못했습니다.",
    "phase_missing": "{role} rep {rep}의 {phase} 구간이 없습니다.",
    "phase_shape": "{role} rep {rep}의 {phase} 구간에는 시작·종료 정수 프레임 2개가 필요합니다.",
    "phase_start": "{role} rep {rep}의 {phase} 구간 시작({start})이 영상 범위(0~{last}) 밖입니다.",
    "phase_end": "{role} rep {rep}의 {phase} 구간 종료({end})가 허용 범위(1~{total}) 밖입니다.",
    "phase_empty": "{role} rep {rep}의 {phase} 구간 시작({start})이 종료({end})보다 앞서지 않습니다.",
    "phase_order": "{role} rep {rep}의 {phase} 구간 시작({start})이 앞 구간 종료({previous})보다 앞입니다.",
    "position_order": "{role} 포지션의 프레임 순서를 확인할 수 없습니다: {positions}.",
    "position_bounds": "{role} 포지션 프레임이 수정 범위({lower}~{last}) 밖입니다: {positions}.",
    "override": "{role} 포지션 수정의 rep 번호나 포지션 이름을 확인할 수 없습니다.",
    "setting_missing": "설정에 필요한 항목({key})이 없습니다.",
    "invalid_data": "분석에 필요한 입력 값을 확인할 수 없습니다 ({fields}).",
}
SKELETON_NOTICE = (
    "0단계 CLI 뼈대: 분할={segment}, 마지막 단계={until}. "
    "영상 처리와 리포트 생성은 이후 단계에서 연결됩니다."
)
