"""Bright 1600x900 slide figures using the installed Korean Noto font."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fontTools.ttLib import TTCollection
ROOT=Path(__file__).resolve().parents[3]
COLORS={'A_existing':'#9AA3AE','B_generic':'#5B6B7F','C_athlete':'#C3CAD3','plate':'#17A08F'}
NAMES={'A_existing':'포즈 A','B_generic':'포즈 B','C_athlete':'포즈 C','plate':'원판'}


def make(v40,raw):
    out=ROOT/'plate_pipeline/results/midterm';raw.mkdir(parents=True,exist_ok=True)
    font=raw/'NotoSansCJK-KR.otf'
    if not font.exists():TTCollection('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').fonts[1].save(font)
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Noto Sans CJK KR','font.size':14,'axes.unicode_minus':False,'figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    comparison=pd.read_csv(out/'comparison.csv')
    def save(fig,name):fig.savefig(out/name,dpi=100,facecolor='white');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(16,9));x=np.arange(3)
    for ax,metric,label in zip(axs.flat,['count_mae','count_exact','obo','event_f1'],['횟수 MAE ↓','정확히 맞음 ↑','OBO (오차 ≤1회) ↑','이벤트 F1 ↑']):
        for j,arm in enumerate(COLORS):
            vals=comparison[comparison.arm==arm][metric].tolist()
            bars=ax.bar(x+(j-1.5)*.2,vals,width=.19,color=COLORS[arm],label=NAMES[arm])
            ax.bar_label(bars,fmt='%.2f',fontsize=10,padding=3)
        ax.set_xticks(x,['기존 시험 52','사용자 시험 9','동일 시험 61']);ax.set_title(label,loc='left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.set_ylim(0,max(comparison[metric])*1.25 if metric=='count_mae' else 1.15)
    axs[0,0].legend(ncol=4,fontsize=10);fig.suptitle('원판 있는 동일 시험 61개 · 횟수 인식 비교',fontsize=24);fig.tight_layout(rect=(0,0,1,.94));save(fig,'counts_summary.png')
    d=pd.read_csv(out/'per_video.csv');b=pd.read_csv(v40/'arms/B_generic/all_test_per_video.csv');b=b[b.key.isin(d.key)]
    fig,axs=plt.subplots(1,2,figsize=(16,9),sharex=True,sharey=True);limit=max(d.gt_count.max(),d.pred_count.max(),b.pred_count.max())+1
    for ax,df,label,color in [(axs[0],b,'포즈 B','#5B6B7F'),(axs[1],d,'원판','#17A08F')]:
        ax.scatter(df.gt_count,df.pred_count,s=90,c=color,alpha=.65,edgecolors='white');ax.plot([0,limit],[0,limit],color='#1F2A5C',ls='--',label='정답과 일치')
        ax.set(xlim=(-.5,limit),ylim=(-.5,limit),xlabel='정답 횟수',ylabel='예측 횟수',title=label);ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend()
    fig.suptitle('영상별 예측 횟수 대 정답 · 겹친 점은 여러 영상',fontsize=24);fig.tight_layout(rect=(0,0,1,.94));save(fig,'counts_scatter.png')
    views=pd.read_csv(out/'by_view.csv').set_index('view').reindex(['옆','비스듬','정면','없음']).dropna(subset=['videos'])
    fig,ax=plt.subplots(figsize=(16,9));bars=ax.bar(views.index,views.count_mae,color='#17A08F',width=.55)
    for bar,(_,r) in zip(bars,views.iterrows()):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'MAE {r.count_mae:.2f}\nn={int(r.videos)}',ha='center',va='bottom',fontsize=18)
    ax.set_ylabel('횟수 MAE');ax.set_ylim(0,max(views.count_mae)*1.3+.1);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.set_title('원판 상자 비율에 따른 횟수 오차 · 자동 조건 태그',loc='left',fontsize=24)
    ax.text(0,-.13,'가로/세로 중앙값: 옆 ≥0.75 · 비스듬 0.4~0.75 · 정면 <0.4 · 측정 없음은 별도',transform=ax.transAxes)
    fig.tight_layout(rect=(0,.06,1,1));save(fig,'by_view.png')
    lines=['# 중간 발표: 원판 있는 동일 시험 61개 조건부 횟수 비교','','| 묶음 | 방법 | MAE ↓ | 정확히 맞음 ↑ | OBO ↑ | event F1 ↑ |','|---|---|---:|---:|---:|---:|']
    for _,r in comparison.iterrows():lines.append(f'| {r.subset} ({int(r.videos)}) | {NAMES[r.arm]} | {r.count_mae:.3f} | {r.count_exact:.3f} | {r.obo:.3f} | {r.event_f1:.3f} |')
    lines+=['','명시된 원판 없음 12개를 뺀 조건부 평가다. 원본 73개 분할은 유지하며 A/B/C 73개 결과는 baseline73_abc.csv에 별도 보존한다.',
      '종목 인식(v40 원본 73개): A 0.863 → B·C 0.973. 원판에는 라벨 종목을 입력했으므로 종목 인식 시험이 아니다.',
      '원판 이벤트는 완료된 rep의 back 시각. back 공란은 완료 이벤트에서 제외했다. 최종 오프라인 결과의 이벤트 시각 오차이며 실제 온라인 통지 지연은 아니다.',
      '각도는 검출된 상자의 자동 태그이며 수작업 촬영 각도 정답이 아니다.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--v40',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);a=p.parse_args();make(a.v40,a.raw)
