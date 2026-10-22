"""S2 adapter contracts, without video/model dependencies."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from feedback import config
from feedback.plate import build_plate_segments, nearest_seq, prepare_videos
from feedback.pose import choose_plate_candidate, extract_pose
from feedback.describe import describe_reps


def synthetic():
    # Up-positive height, with an extended far pause; frames have intentionally non-fps times.
    heights = [0, 0, -.2, -.4, -.6, -.8, -1, -1, -.8, -.6, -.4, -.2, 0, 0]
    rows = [{'seq': i, 't': 10 + i * .13, 'plate': {
        'used': True, 'box': [0, 0, 45, 45], 'center': [0, -100*h], 'size_px': [45,45]}}
        for i, h in enumerate(heights)]
    pose = {'total': len(rows), 'fps': 30, 'w': 100, 'h': 100,
            'lms': np.zeros((len(rows), 33, 4)), 'valid': np.zeros(len(rows), bool)}
    value = {'start': rows[1]['t'], 'far_in': rows[6]['t'],
             'far_out': rows[7]['t'], 'back': rows[12]['t']}
    result = {'sets': [{'rep_values': [value]}]}
    settings = {'video': 'video.mp4', 'view': 'side'}
    return pose, settings, rows, result


@pytest.mark.parametrize('exercise', ['squat', 'bench', 'deadlift'])
def test_plate_positions_phases_and_alias(exercise):
    pose, settings, rows, result = synthetic()
    seg = build_plate_segments(pose, settings, exercise, rows, result)
    rep = seg['reps'][0]
    assert list(rep['positions'].values()) == [1, 4, 6, 10, 12]
    assert rep['phases'] == {'down': [1, 6], 'bottom': [6, 8], 'up': [8, 13]}
    assert seg['source'] == 'plate'
    assert seg['exercise'] == ('benchpress' if exercise == 'bench' else exercise)
    assert nearest_seq(rows, rows[4]['t'] + .02) == 4


def test_reverse_far_times_and_missing_back_across_sets():
    pose, settings, rows, result = synthetic()
    value = result['sets'][0]['rep_values'][0]
    value['far_in'], value['far_out'] = value['far_out'], value['far_in']
    early = {**value, 'start': 1, 'back': None}
    result['sets'].append({'rep_values': [early]})
    seg = build_plate_segments(pose, settings, 'squat', rows, result)
    assert seg['plate_rep_count'] == 2
    assert seg['excluded_reps'] == [{'rep': 1, 'reason': 'rep 끝 시점 없음'}]
    rep = seg['reps'][0]
    assert rep['rep'] == 2
    assert rep['phases']['bottom'] == [6, 7]
    assert rep['phases']['down'][1] == rep['phases']['bottom'][0]
    assert rep['phases']['bottom'][1] == rep['phases']['up'][0]


def test_count_mismatch_stops():
    pose, settings, rows, result = synthetic()
    with pytest.raises(ValueError, match='포즈 프레임'):
        build_plate_segments(pose, settings, 'squat', rows[:-1], result)


def person(x, scale=1):
    p = np.ones((33,4), float)
    p[:,0] = x + np.linspace(0,.1*scale,33)
    p[:,1] = np.linspace(.1,.9*scale,33)
    return p


def test_select_lifter_plate_near_second_and_no_plate_fallback():
    first, second = person(.1), person(.7)
    row = {'plate': {'used': True, 'box': [700,400,800,500]}}
    selected, center = choose_plate_candidate([first,second], row, None, 'squat', (1000,1000))
    np.testing.assert_array_equal(selected, second)
    assert center[0] > 700
    # Same size: continuity chooses the second candidate when plate is absent.
    selected, _ = choose_plate_candidate([first,second], {}, center, 'squat', (1000,1000))
    np.testing.assert_array_equal(selected, second)
    # No plate/no history: the largest candidate wins.
    big = person(.1,1.5)
    selected, _ = choose_plate_candidate([second,big], {}, None, 'squat', (1000,1000))
    np.testing.assert_array_equal(selected, big)
    assert choose_plate_candidate([], {}, center, 'squat', (1000,1000))[0] is None


def test_fix_preserves_exact_average_rate(monkeypatch, tmp_path):
    from tools import check_video
    calls = []
    monkeypatch.setattr(check_video, 'ffprobe', lambda p: {'avg_rate':'60000/1001'})
    monkeypatch.setattr(check_video.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    check_video.fix(Path('rotated.mov'), tmp_path)
    cmd = calls[-1]
    assert cmd[cmd.index('-r')+1] == '60000/1001'
    for option, value in [('-vsync','cfr'),('-c:v','libx264'),('-pix_fmt','yuv420p'),('-metadata:s:v','rotate=0')]:
        assert cmd[cmd.index(option)+1] == value
    assert '-an' in cmd and '-noautorotate' not in cmd
    check_video.fix(Path('rotated.mov'), tmp_path, fps=30)
    assert calls[-1][calls[-1].index('-r')+1] == '30'


def test_prepare_uses_one_copy_for_both_roles(monkeypatch, tmp_path):
    from tools import check_video
    calls = []
    monkeypatch.setattr(check_video, 'ffprobe', lambda p: {'codec':'hevc','rotation':90})
    monkeypatch.setattr(check_video, 'fix', lambda p, d: calls.append((p,d)) or d/'copy.mp4')
    settings = {role:{'video':'same.mov'} for role in ('user','ref')}
    prepare_videos(settings,tmp_path)
    assert len(calls) == 1
    assert settings['user']['video'] == settings['ref']['video']
    monkeypatch.setattr(check_video, 'ffprobe', lambda p: {'codec':'h264','pix_fmt':'yuv420p','r_fps':60,'avg_fps':60})
    prepare_videos(settings,tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize('metric,unit,positive,negative', [
    ('elbow_angle','deg','팔꿈치를 레퍼런스보다 12° 더 폈습니다','팔꿈치를 레퍼런스보다 12° 더 굽혔습니다'),
    ('lockout_angle_min','deg','팔꿈치를 레퍼런스보다 12° 더 폈습니다','팔꿈치를 레퍼런스보다 12° 더 굽혔습니다'),
    ('wrist_elbow_x_diff','ratio','손목과 팔꿈치의 가로 거리가 레퍼런스보다 몸통 길이의 12%만큼 멉니다','손목과 팔꿈치의 가로 거리가 레퍼런스보다 몸통 길이의 12%만큼 가깝습니다')])
@pytest.mark.parametrize('direction', ['+','-'])
def test_bench_sentences(metric, unit, positive, negative, direction):
    cell = {'where':'BOTTOM','metric':metric,'unit':unit,'diff':(.12 if unit=='ratio' else 12)*(1 if direction=='+' else -1)}
    rep = {'rep':1,'excluded':False,'goal':{'type':'single','where':'BOTTOM','metric':metric,'dir':direction},
           'cells':[cell],'tempo_goal':None}
    sentence = describe_reps({'exercise':'benchpress','reps':[rep]})[0]['goal_sentence']
    assert sentence == '가슴에서 ' + (positive if direction=='+' else negative) + '.'
    assert not any(word in sentence for word in config.BANNED)


def test_dead_names_and_no_depth_group():
    from feedback.select import select_goals
    assert list(config.position_labels('deadlift').values()) == ['시작(바닥)','상승 중간','락아웃','하강 중간','끝(바닥)']
    assert list(config.phase_labels('deadlift').values()) == ['상승','위','하강']
    rep = {'rep':1,'excluded':False,'cells':[{'where':'BOTTOM','metric':m,'status':'ok','k':2,'dir':'-',
           'diff':-36,'unit':'deg','color':'red'} for m in ('knee_angle','hip_angle')], 'tempo':[], 'tempo_goal':None}
    findings = select_goals({'exercise':'deadlift','reps':[rep]})
    assert findings['reps'][0]['goal']['type'] == 'single'
    sentence = describe_reps(findings)[0]['goal_sentence']
    assert sentence == '락아웃에서 무릎을 레퍼런스보다 36° 더 굽혔습니다.'
    assert not any(word in sentence for word in config.BANNED)
    assert 'lockout_angle_min' in config.ADVISORY_METRICS['benchpress']


def test_cache_selection_is_part_of_identity(monkeypatch, tmp_path):
    from feedback import pose as module
    import offline_compare
    import export_overlay
    video = tmp_path/'video.mp4'; video.touch()
    calls=[]
    monkeypatch.setattr(export_overlay,'video_info',lambda p:{'fps':30,'w':100,'h':100})
    monkeypatch.setattr(offline_compare,'extract_sequence',lambda *a:calls.append('first') or [None])
    monkeypatch.setattr(module,'extract_plate_sequence',lambda *a:calls.append('plate') or [None])
    cache=tmp_path/'pose.npz'
    assert not extract_pose(video,cache,'squat')['cache_hit']
    assert extract_pose(video,cache,'squat')['cache_hit']
    rows=[{'seq':0}]
    assert not extract_pose(video,cache,'squat',plate_rows=rows)['cache_hit']
    assert extract_pose(video,cache,'squat',plate_rows=rows)['cache_hit']
    assert not extract_pose(video,cache,'squat')['cache_hit']
    assert calls == ['first','plate','first']


def test_all_goal_metrics_have_templates_and_3d_arm_mapping():
    for exercise, names in config.DISPLAY_METRICS.items():
        for name in names:
            if name not in config.ADVISORY_METRICS[exercise] and config.DELTA_THRESHOLDS[exercise][name] > 0:
                assert set(config.METRIC_TEMPLATES[name]) == {'+','-'}
    assert config.POSE3D_METRIC_JOINTS['elbow_angle'] == {'right':[14,15,16],'left':[11,12,13]}
    assert config.POSE3D_METRIC_JOINTS['wrist_elbow_x_diff'] == {'right':[15,16],'left':[12,13]}


def test_excluded_reference_stays_excluded_in_plate_report():
    from feedback.compare import compare_reps
    from feedback.select import select_goals
    positions = {p: {'values': {m: None for m in config.DISPLAY_METRICS['squat']}} for p in config.POSITIONS}
    rep = {'rep':3, 'positions':positions, 'detect':dict.fromkeys(config.PHASES,1),
           'excluded':False,'exclude_reason':None,'tempo':{'down_s':1,'bottom_s':.1,'up_s':1,'ratio':1}}
    metrics = {'exercise':'squat','reps':[rep]}
    with pytest.raises(ValueError):
        compare_reps(metrics,metrics,3)
    findings = select_goals(compare_reps(metrics,metrics,3,allow_excluded_reference=True))
    assert findings['reps'][0]['excluded']
    assert '측정 가능한 칸 0%' in findings['reps'][0]['exclude_reason']
    assert findings['reps'][0]['goal'] is None


def test_ffprobe_detects_variable_cadence_even_with_equal_rates(monkeypatch):
    import json
    from types import SimpleNamespace
    from tools import check_video
    monkeypatch.setattr(check_video.shutil,'which',lambda _: '/bin/ffprobe')
    payload={'streams':[{'codec_name':'h264','avg_frame_rate':'30/1','r_frame_rate':'30/1',
                        'side_data_list':[{'rotation':90}]}],
             'packets':[{'pts_time':str(t)} for t in [0,.033333,.066667,.133333]]}
    monkeypatch.setattr(check_video.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=json.dumps(payload)))
    info=check_video.ffprobe(Path('variable.mp4'))
    assert info['variable_fps'] and info['rotation'] == 90


@pytest.mark.parametrize('phase,direction,expected',[
    ('down_s','+','상승이 레퍼런스보다 0.5초 느립니다 (상승:하강 1.0:1, 레퍼런스 1.0:1).'),
    ('up_s','-','하강이 레퍼런스보다 0.5초 빠릅니다 (상승:하강 1.0:1, 레퍼런스 1.0:1).'),
    ('bottom_s','+','위에서 레퍼런스보다 0.5초 더 머물렀습니다.')])
def test_dead_tempo_sentences(phase,direction,expected):
    rep={'rep':1,'excluded':False,'goal':{'type':'none'},'cells':[],
         'tempo_ratio':{'user':1,'ref':1},'tempo_goal':{'metric':phase,'dir':direction,'diff':.5}}
    sentence=describe_reps({'exercise':'deadlift','reps':[rep]})[0]['tempo_sentence']
    assert sentence == expected
    assert not any(word in sentence for word in config.BANNED)


def test_plate_api_uses_pose_false_box_model_and_device(monkeypatch, tmp_path):
    from feedback.plate import analyze_video
    from core.io.rows import write_rows
    from server import jobs
    calls=[]
    def fake_process(path, exercise, **kwargs):
        calls.append((path,exercise,kwargs))
        tmp_path.mkdir(exist_ok=True)
        write_rows(tmp_path/'rows.jsonl.gz',[{'seq':0,'t':0}])
        (tmp_path/'result.json').write_text('{"sets":[]}')
    monkeypatch.setattr(jobs,'process_video',fake_process)
    rows,result=analyze_video({'video':'one.mp4','plate_box':[1,2,3,4]},'benchpress',tmp_path,
                              model=Path('plate.onnx'),device='cpu')
    assert calls == [('one.mp4','bench',{'init_box':[1,2,3,4],'model':'plate.onnx','device':'cpu',
                                        'pose':False,'out_dir':tmp_path})]
    assert rows == [{'seq':0,'t':0}] and result == {'sets':[]}


def test_plate_cli_shares_pose_but_preserves_set_detection_exclusion(monkeypatch,tmp_path):
    import run_demo
    from feedback import plate,pose as module
    pose,settings,rows,result=synthetic()
    settings={'exercise':'squat','user':settings,'ref':{**settings,'rep':1}}
    calls=[]
    def fake_pose(video,cache,exercise,**kwargs):
        calls.append(('pose',video))
        cache.write_bytes(b'shared-cache')
        return {**pose,'video':str(video.resolve()),'cache_hit':False}
    def fake_analyze(*args,**kwargs):
        calls.append(('plate',args[0]['video']))
        return rows,result
    monkeypatch.setattr(module,'extract_pose',fake_pose)
    monkeypatch.setattr(plate,'analyze_video',fake_analyze)
    args=run_demo.build_parser().parse_args(['--out',str(tmp_path)])
    results=run_demo._run_segments(settings,args,{},'plate',validate=True)
    assert [kind for kind,_ in calls] == ['plate','pose','plate']
    assert (tmp_path/'pose_user.npz').read_bytes() == (tmp_path/'pose_ref.npz').read_bytes()
    measured=run_demo._run_metrics(results,tmp_path)
    for role in ['user','ref']:
        assert measured[role]['reps'][0]['excluded']
        assert '영상의 관절 검출률' in measured[role]['reps'][0]['exclude_reason']
