"""Pipeline diagram v2 (1600x900): boxes tagged by presentation part (AI = recognition, 보조 = correction/compare)."""
NAVY, TEAL, ORANGE, GREY, CARD, INK, MUTED = "#1F2A5C", "#17A08F", "#E2862E", "#B6C0D2", "#EDF1FA", "#1E2433", "#5B6475"
W, H = 1600, 900
els = []


def tag(x, y, kind):
    if kind == "AI":
        els.append(f'<rect x="{x}" y="{y - 10}" width="34" height="20" rx="10" fill="{NAVY}"/>')
        els.append(f'<text x="{x + 17}" y="{y + 5}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#FFFFFF">AI</text>')
    else:
        els.append(f'<rect x="{x}" y="{y - 10}" width="40" height="20" rx="10" fill="#FFFFFF" stroke="{MUTED}" stroke-width="1.5"/>')
        els.append(f'<text x="{x + 20}" y="{y + 5}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{INK}">보조</text>')


def box(x, y, w, h, title, sub=None, color=NAVY, fill="#FFFFFF", tcol=INK, kind=None, dash=False, tsize=20):
    d = ' stroke-dasharray="8 6"' if dash else ""
    els.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{color}" stroke-width="2.5"{d}/>')
    cy = y + h / 2 + (-8 if sub else 7)
    els.append(f'<text x="{x + w / 2}" y="{cy}" text-anchor="middle" font-size="{tsize}" font-weight="700" fill="{tcol}">{title}</text>')
    if sub:
        els.append(f'<text x="{x + w / 2}" y="{cy + 25}" text-anchor="middle" font-size="14.5" fill="{MUTED if fill == "#FFFFFF" else tcol}">{sub}</text>')
    if kind:
        tag(x + 12, y, kind)


def arrow(x1, y1, x2, y2, color=INK, dash=False):
    d = ' stroke-dasharray="7 6"' if dash else ""
    els.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.2"{d} marker-end="url(#ah)"/>')


def path(pts, color=INK, dash=False, head=True):
    d = ' stroke-dasharray="7 6"' if dash else ""
    p = " ".join(f"{x},{y}" for x, y in pts)
    m = ' marker-end="url(#ah)"' if head else ""
    els.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2.2"{d}{m}/>')


def lane(x, y, w, h, label, color):
    els.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{CARD}"/>')
    els.append(f'<rect x="{x}" y="{y}" width="8" height="{h}" rx="4" fill="{color}"/>')
    els.append(f'<text x="{x + 24}" y="{y + 30}" font-size="17" font-weight="700" fill="{color}">{label}</text>')


def note(x, y, s, color):
    els.append(f'<text x="{x}" y="{y}" font-size="14" fill="{color}">{s}</text>')


# header + legend
els.append(f'<text x="60" y="62" font-size="32" font-weight="900" fill="{NAVY}">전체 파이프라인: 영상 두 개 → 기록과 피드백</text>')
els.append(f'<text x="60" y="94" font-size="17" fill="{MUTED}">원판이 1차 신호 (횟수 · 깊이 · 템포 · 속도), 포즈는 보조 (자세 · 종목 · 3D)</text>')
tag(1130, 58, "AI"); els.append(f'<text x="1172" y="63" font-size="15" fill="{INK}">인식 (검출 · 추적 · 포즈 · 종목 · 횟수)</text>')
tag(1130, 88, "보조"); els.append(f'<text x="1178" y="93" font-size="15" fill="{INK}">보정 · 비교 (각도 · 크기 · 정렬 · 판정)</text>')

# inputs
box(60, 160, 200, 66, "사용자 영상", "옆에서 찍은 세트")
box(60, 240, 200, 66, "기준 영상", "따라 하고 싶은 세트")
box(60, 340, 200, 80, "영상 확인 · 변환", "회전 · H.264 · 원래 fps", color=GREY)
path([(60, 193), (38, 193), (38, 380), (58, 380)])
els.append(f'<line x1="60" y1="273" x2="38" y2="273" stroke="{INK}" stroke-width="2.2"/>')
path([(260, 380), (282, 380), (282, 238), (323, 238)])
path([(282, 380), (282, 472), (323, 472)])

# lane A: plate
lane(300, 140, 840, 200, "원판 경로 · 1차 신호", TEAL)
box(325, 196, 180, 84, "원판 검출", "YOLOX-Tiny (학습)", TEAL, kind="AI")
box(530, 196, 180, 84, "원판 추적", "KLT + 재검출", TEAL, kind="AI")
box(735, 196, 180, 84, "각도 보정 · cm", "원판 타원 · 지름", TEAL, kind="보조")
box(940, 196, 180, 84, "rep 상태기계", "깊이 · 템포 · 속도", TEAL, kind="AI")
arrow(505, 238, 528, 238); arrow(710, 238, 733, 238); arrow(915, 238, 938, 238)
# user input: plate diameter (and load)
els.append(f'<rect x="745" y="148" width="160" height="28" rx="14" fill="#FFFFFF" stroke="{TEAL}" stroke-width="1.8"/>')
els.append(f'<text x="825" y="167" text-anchor="middle" font-size="14" font-weight="700" fill="{TEAL}">입력: 원판 지름</text>')
arrow(825, 176, 825, 194, TEAL)

# lane B: pose
lane(300, 380, 840, 220, "포즈 경로 · 보조 신호", ORANGE)
box(325, 430, 180, 84, "포즈 추정", "MediaPipe 33점", ORANGE, kind="AI")
box(530, 430, 180, 84, "리프터 고르기", "원판에 가까운 사람", ORANGE, kind="AI")
box(735, 430, 180, 84, "종목 인식", "ST-GCN", ORANGE, kind="AI")
box(940, 430, 180, 84, "3D 자세 · 지표", "실제 크기 · 관절 각도", ORANGE, kind="보조")
arrow(505, 472, 528, 472); arrow(710, 472, 733, 472)
path([(620, 514), (620, 562), (1030, 562), (1030, 516)], ORANGE)

# cross links between lanes
path([(620, 280), (620, 428)], TEAL, dash=True); note(630, 328, "원판 위치", TEAL)
path([(825, 430), (825, 312), (990, 312), (990, 282)], ORANGE, dash=True); note(835, 305, "종목", ORANGE)
path([(1075, 280), (1075, 428)], TEAL, dash=True); note(1083, 360, "rep 시각", TEAL)

# compare column
box(1180, 160, 360, 88, "바 높이로 rep 정렬", "템포가 달라도 같은 동작끼리", NAVY, fill=NAVY, tcol="#FFFFFF", kind="보조")
box(1180, 290, 360, 88, "세트 판정 · rep 북마크", "세트 평균 k·SE · 혼자 보이는 rep만 표시", NAVY, fill=NAVY, tcol="#FFFFFF", kind="보조")
box(1180, 420, 360, 88, "문장 만들기", "판정어 없이 차이만", NAVY, fill=NAVY, tcol="#FFFFFF", kind="보조")
path([(1120, 238), (1150, 238), (1150, 204), (1178, 204)])
path([(1120, 472), (1155, 472), (1155, 334), (1178, 334)])
arrow(1360, 248, 1360, 288); arrow(1360, 378, 1360, 418)

# outputs
lane(60, 650, 1480, 210, "출력", NAVY)
box(110, 700, 440, 116, "비교 리포트", "나란히 재생 · 3D 겹쳐 보기 · 북마크 · 문장", tsize=21)
box(590, 700, 420, 116, "운동 기록", "날짜별 세트 · 횟수 · 깊이 · 속도 → 리포트", tsize=21)
box(1050, 700, 440, 116, "피드백 영상 · 실시간 화면", "다음 단계", color=GREY, dash=True, tcol=MUTED, tsize=21)
path([(1360, 508), (1360, 622), (330, 622), (330, 698)])
path([(800, 622), (800, 698)])
path([(1270, 622), (1270, 698)], GREY, dash=True)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Noto Sans KR, Noto Sans CJK KR, Malgun Gothic, sans-serif">'
       f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(els) + '</svg>')
open("pipeline_v2.svg", "w", encoding="utf-8").write(svg)
open("pipeline_v2.html", "w", encoding="utf-8").write(f'<!doctype html><meta charset="utf-8"><body style="margin:0">{svg}</body>')
