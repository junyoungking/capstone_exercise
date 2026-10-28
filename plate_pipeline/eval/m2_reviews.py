"""Reclassify existing reviewed wrong-person samples by completed plate rep interval."""
from pathlib import Path
import argparse,json
from eval.baseline import write,sha
from feedback.plate import plate_rep_mask
from core.io.rows import read_rows

# Only confirmed wrong-person frames, not later 'fixed' or 'correct' frame mentions.
WRONG={
 'M0':{'benchpress/stu10_11.mp4':[359,719,1078,1438],'benchpress/test153.mp4':[0,239,299],
       'squat/stu6_11_2.mp4':[127,159],'squat/stu6_11_3.mp4':[0,38],'deadlift/GB_S04_R03.mp4':[36],
       'team9/benchpress/stu9_10.mp4':[243,401,558,740,904,1085]},
 'M1 lightweight':{'squat/val1235.mp4':[52,232],'benchpress/stu10_11.mp4':[0],'benchpress/test153.mp4':[60],
                    'squat/stu6_11_2.mp4':[95],'team9/benchpress/stu9_10.mp4':[401,558,740]},
 'M1 balanced':{'squat/val1235.mp4':[0,105,179],'benchpress/stu10_11.mp4':[0]},
 'M1b n3':{'squat/val1235.mp4':[0],'benchpress/stu10_11.mp4':[0],'squat/stu6_11_2.mp4':[95]},
 'M1b n5':{'squat/val1235.mp4':[0],'benchpress/stu10_11.mp4':[0,1438],'squat/stu6_11_2.mp4':[95]},
 'M1b n10':{'squat/val1235.mp4':[0],'benchpress/stu10_11.mp4':[0],'squat/stu6_11_2.mp4':[95]},
 'M1b tiny_m':{'squat/val1235.mp4':[52,105,179],'benchpress/stu10_11.mp4':[0],'benchpress/test153.mp4':[60],
              'squat/stu6_11_2.mp4':[95],'deadlift/LLK_S01_R03.mp4':[59],'team9/benchpress/stu9_10.mp4':[558]},
 'M1b bench_guard':{'benchpress/stu10_11.mp4':[0]},
}


def annotate(name,root,wrong=None):
    report=json.loads((root/'baseline.json').read_text());reviews=json.loads((root/'lifter_review.json').read_text())
    wrong=WRONG[name] if wrong is None else wrong
    assert {k for k,v in reviews.items() if v.startswith('wrong')}==set(wrong)
    entries=[]
    for v in report['videos']:
        key=v['key'];source=root/'raw'/key
        stage=source/('pose_plate' if (source/'pose_plate').exists() else 'plate')
        rows=read_rows(stage/'rows.jsonl.gz');result=json.loads((stage/'result.json').read_text())
        mask=plate_rep_mask(rows,result);frames=wrong.get(key,[])
        assert set(frames)<=set(v['image_frames']),(name,key,frames,v['image_frames'])
        entries.append(dict(key=key,review=reviews[key],image=v['image'],sample_frames=v['image_frames'],
                            sampled_inside=[int(f) for f in v['image_frames'] if mask[f]],
                            sampled_outside=[int(f) for f in v['image_frames'] if not mask[f]],
                            wrong_inside=[f for f in frames if mask[f]],wrong_outside=[f for f in frames if not mask[f]]))
    return dict(source=str(root),review_sha256=sha(root/'lifter_review.json'),videos=entries,
                wrong_inside_videos=sum(bool(v['wrong_inside']) for v in entries),
                wrong_outside_videos=sum(bool(v['wrong_outside']) for v in entries),
                wrong_union_videos=sum(bool(v['wrong_inside'] or v['wrong_outside']) for v in entries))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--m0',type=Path,required=True)
    p.add_argument('--m1',type=Path,required=True);p.add_argument('--m1b',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    roots={'M0':a.m0,**{'M1 '+m:a.m1/m for m in ('lightweight','balanced')},
           **{'M1b '+m:a.m1b/m for m in ('n3','n5','n10','tiny_m','bench_guard')}}
    result={name:annotate(name,path) for name,path in roots.items()};write(a.out,result)
    for name,v in result.items():print(name,v['wrong_inside_videos'],v['wrong_outside_videos'])


if __name__=='__main__':main()
