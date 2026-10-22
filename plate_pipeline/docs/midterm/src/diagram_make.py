"""Pipeline diagram: boxes and lines, slide-ready (1600x900)."""
NAVY, TEAL, ORANGE, GREY, CARD, INK, MUTED = "#1F2A5C", "#17A08F", "#E2862E", "#B6C0D2", "#EDF1FA", "#1E2433", "#5B6475"
W, H = 1600, 900
els = []
def box(x, y, w, h, title, sub=None, color=NAVY, fill="#FFFFFF", tcol=INK, bold=True):
    els.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{color}" stroke-width="2.5"/>')
    cy = y + h / 2 + (-9 if sub else 7)
    els.append(f'<text x="{x+w/2}" y="{cy}" text-anchor="middle" font-size="21" font-weight="{700 if bold else 500}" fill="{tcol}">{title}</text>')
    if sub:
        els.append(f'<text x="{x+w/2}" y="{cy+26}" text-anchor="middle" font-size="15.5" fill="{MUTED if fill=="#FFFFFF" else tcol}">{sub}</text>')
    return (x, y, w, h)
def arrow(x1, y1, x2, y2, color=INK, dash=False):
    d = ' stroke-dasharray="7 6"' if dash else ""
    els.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.2"{d} marker-end="url(#ah)"/>')
def path(pts, color=INK, dash=False):
    d = ' stroke-dasharray="7 6"' if dash else ""
    p = " ".join(f"{x},{y}" for x, y in pts)
    els.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2.2"{d} marker-end="url(#ah)"/>')
def lane(x, y, w, h, label, color):
    els.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{CARD}" stroke="none"/>')
    els.append(f'<rect x="{x}" y="{y}" width="8" height="{h}" rx="4" fill="{color}"/>')
    els.append(f'<text x="{x+24}" y="{y+30}" font-size="17" font-weight="700" fill="{color}">{label}</text>')

els.append(f'<text x="60" y="64" font-size="32" font-weight="900" fill="{NAVY}">통합 파이프라인: 영상 입력부터 피드백까지</text>')
els.append(f'<text x="60" y="96" font-size="17" fill="{MUTED}">원판이 1차 신호(횟수·템포·속도), 포즈는 보조(자세·종목·골격 그림)</text>')

# input column
box(60, 170, 200, 70, "사용자 영상", "옆에서 찍은 세트")
box(60, 260, 200, 70, "기준 영상", "사용자가 고른 영상")
box(60, 380, 200, 90, "영상 확인·변환", "회전 · H.264 · 원래 fps", color=GREY)
path([(60, 205), (38, 205), (38, 425), (58, 425)]); els.append(f'<line x1="60" y1="295" x2="38" y2="295" stroke="{INK}" stroke-width="2.2"/>')

# lane A: plate
lane(310, 140, 1000, 210, "원판 경로 · 1차 신호", TEAL)
a1 = box(340, 190, 190, 80, "원판 검출", "YOLOX-Tiny", TEAL)
a2 = box(560, 190, 190, 80, "원판 추적", "KLT + 재검출", TEAL)
a3 = box(780, 190, 190, 80, "바 높이 신호", "cm 환산(원판 450 mm)", TEAL)
a4 = box(1000, 190, 280, 80, "rep 상태기계", "rep 나누기 · 세트 닫기", TEAL)
arrow(530, 230, 558, 230); arrow(750, 230, 778, 230); arrow(970, 230, 998, 230)

# lane B: pose
lane(310, 380, 1000, 210, "포즈 경로 · 보조", ORANGE)
b1 = box(340, 430, 190, 80, "포즈 추정", "MediaPipe 33점", ORANGE)
b2 = box(560, 430, 190, 80, "리프터 고르기", "원판에 가까운 손목", ORANGE)
b3 = box(780, 430, 190, 80, "자세 지표", "포지션 5개 × 관절 각도", ORANGE)
b4 = box(1000, 430, 280, 80, "종목 인식", "ST-GCN + 원판 방향", ORANGE)
arrow(530, 470, 558, 470); arrow(750, 470, 778, 470)
path([(655, 510), (655, 545), (1140, 545), (1140, 512)], ORANGE)

# input -> lanes
path([(260, 425), (290, 425), (290, 230), (338, 230)])
path([(290, 425), (290, 470), (338, 470)])
# cross links
path([(1140, 270), (1140, 320), (875, 320), (875, 428)], TEAL, dash=True)   # rep positions -> pose metrics
els.append(f'<text x="890" y="312" font-size="14.5" fill="{TEAL}">rep 포지션</text>')
path([(1240, 430), (1240, 272)], ORANGE, dash=True)                           # exercise -> rep machine
els.append(f'<text x="1250" y="360" font-size="14.5" fill="{ORANGE}">종목</text>')
path([(645, 270), (645, 300), (600, 300), (600, 428)], TEAL, dash=True)       # plate box -> lifter select
els.append(f'<text x="612" y="362" font-size="14.5" fill="{TEAL}">원판 위치</text>')

# judge + output
c1 = box(1360, 170, 200, 110, "비교 판정", "세트 단위 · k·SE", NAVY, fill=NAVY, tcol="#FFFFFF")
c2 = box(1360, 320, 200, 100, "문장 만들기", "템플릿 · 판정어 없음", NAVY, fill=NAVY, tcol="#FFFFFF")
path([(1280, 230), (1358, 230)])
path([(1280, 470), (1320, 470), (1320, 250), (1358, 250)])
arrow(1460, 280, 1460, 318)

lane(60, 640, 1500, 200, "출력", NAVY)
o1 = box(100, 690, 330, 110, "세트 기록", "횟수 · 깊이 · 템포 · 속도 · 속도 저하")
o2 = box(470, 690, 330, 110, "비교 리포트 페이지", "나란히 보기 · 템포 · 포지션 표 · 3D")
o3 = box(840, 690, 330, 110, "피드백 영상", "바 경로 · 횟수 · 골격 · 느린 재생")
o4 = box(1210, 690, 310, 110, "실시간 두 번째 화면", "큰 횟수 · 방금 rep 속도")
path([(1460, 420), (1460, 600), (265, 600), (265, 688)])
for x in (635, 1005, 1365):
    path([(x, 600), (x, 688)])

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Noto Sans KR, Noto Sans CJK KR, Malgun Gothic, sans-serif">'
       f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(els) + '</svg>')
open("pipeline.svg", "w", encoding="utf-8").write(svg)
open("pipeline.html", "w", encoding="utf-8").write(f'<!doctype html><meta charset="utf-8"><body style="margin:0">{svg}</body>')
