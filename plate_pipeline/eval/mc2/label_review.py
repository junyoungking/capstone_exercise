"""Create empty human review sheets; no automatic label adjudication."""
import csv,json,sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.io.rows import read_rows


def frame(cap, rows, seconds, origin, width=400, height=310):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, seconds)*1000)
    ok, im=cap.read()
    if not ok:
        return np.zeros((height,width,3),dtype=np.uint8)
    r=min(rows,key=lambda x:abs(x['t']-origin-max(0,seconds)))
    box=r.get('plate',{}).get('box')
    if box:
        x0,y0,x1,y1=map(int,box);cv2.rectangle(im,(x0,y0),(x1,y1),(0,0,255),max(2,im.shape[1]//400))
    s=min(width/im.shape[1],height/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*s),round(im.shape[0]*s)))
    canvas=np.zeros((height,width,3),np.uint8);canvas[:im.shape[0],:im.shape[1]]=im
    return canvas


def run():
    fails=json.loads((ROOT/'results/mc/failures.json').read_text())
    bykey={r['key']:r for r in json.loads((ROOT/'results/mc/baseline.json').read_text())}
    chosen={}
    for event in fails:
        if event.get('scope')=='inside' and event.get('type') in ('아','마') and event.get('direction'):
            chosen.setdefault(event['key'],{'FN':[],'FP':[]})[event['direction']].append(event['time_s'])
    out=ROOT/'eval_out/mc2/label_review';out.mkdir(parents=True,exist_ok=True)
    lines=[]
    for key,ev in chosen.items():
        b=bykey[key]; rows=read_rows(Path(b['result_path'])/'rows.jsonl.gz');cap=cv2.VideoCapture(b['video'])
        unused=ev['FP'][:];pairs=[]
        for gt in ev['FN']:
            pred=min(unused,key=lambda p:abs(p-gt)) if unused else None
            if pred is not None:unused.remove(pred)
            pairs.append((gt,pred))
        pairs += [(None,p) for p in unused]
        for n,(gt,pred) in enumerate(pairs,1):
            panels=[]
            for name,t in [('GT',gt),('Prediction',pred)]:
                parts=[]
                for delta in (-.5,0,.5):
                    im=frame(cap,rows,t+delta,b['origin']) if t is not None else np.zeros((310,400,3),np.uint8)
                    label=f'{name} {"missing" if t is None else f"{t:.3f} {delta:+.1f}s"}'
                    cv2.putText(im,label,(6,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1,cv2.LINE_AA);parts.append(im)
                panels.append(np.hstack(parts))
            image=np.vstack(panels);filename=key.replace('/','__').replace('.mp4','')+f'_{n:02}.jpg';cv2.imwrite(str(out/filename),image,[int(cv2.IMWRITE_JPEG_QUALITY),88])
            lines.append({'영상':key,'종목':b['exercise'],'GT 시각':gt,'예측 시각':pred,'판정':'','메모':'','이미지':filename})
        cap.release()
    with (out/'review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['영상','종목','GT 시각','예측 시각','판정','메모','이미지']);w.writeheader();w.writerows(lines)
    (out/'README.txt').write_text('rep 끝 = 바가 시작 위치로 돌아온 순간 (스쿼트·벤치는 위, 데드는 바닥)\n판정: GT 맞음 / 예측 맞음 / 둘 다 아님. 사람의 판단 전까지 판정 칸은 비워 둡니다.\n빨간 상자는 당시 추적된 원판이며 검출 실패일 수 있습니다.\n',encoding='utf-8')
    print(len(lines),'review sheets:',out)

if __name__=='__main__':run()
