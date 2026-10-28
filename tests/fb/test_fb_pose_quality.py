"""S3a: plate support windows and confidence of the joints actually used."""
from copy import deepcopy

import numpy as np
import pytest

from feedback import gate
from feedback.metrics import build_metrics
from test_fb_metrics import _pose, _segments


def plate_segments(total=12):
    segments=_segments(total)
    segments['source']='plate'
    return segments


def test_detection_counts_union_of_inclusive_plate_rep_intervals():
    pose=_pose(total=100)
    pose['valid'][:]=False;pose['valid'][40:51]=True
    segments=plate_segments(100)
    segments['reps'][0]['positions'].update(START=40,LOCKOUT=50)
    # An overlapping rep must not double-count the denominator.
    second=deepcopy(segments['reps'][0]);second['rep']=2
    second['positions'].update(START=45,LOCKOUT=50)
    segments['reps'].append(second)
    result=gate.check_detection('user',pose,segments)
    assert (result['detected'],result['total'],result['ratio']) == (11,11,1)
    with pytest.raises(gate.GateFailure):gate.check_detection('user',pose)


def test_plate_detection_keeps_the_sixty_percent_boundary():
    pose=_pose(total=30);pose['valid'][:]=False;pose['valid'][10:16]=True
    segments=plate_segments(30);segments['reps'][0]['positions'].update(START=10,LOCKOUT=19)
    assert gate.check_detection('user',pose,segments)['ratio'] == .6
    pose['valid'][15]=False
    with pytest.raises(gate.GateFailure):gate.check_detection('user',pose,segments)


def test_short_plate_phase_uses_its_position_window_but_tempo_is_unchanged():
    pose=_pose();pose['valid'][6]=False
    segments=plate_segments()
    segments['reps'][0]['phases']={'down':[0,6],'bottom':[6,7],'up':[7,12]}
    result=build_metrics(pose,segments)['reps'][0]
    assert result['detect']['bottom'] == .8
    assert result['tempo']['bottom_s'] == .1
    legacy=deepcopy(segments);legacy['source']='labels'
    assert build_metrics(pose,legacy)['reps'][0]['detect']['bottom'] == 0


def test_short_phase_rejects_only_two_detected_frames_in_window():
    pose=_pose();pose['valid'][:]=False;pose['valid'][4:6]=True
    segments=plate_segments()
    segments['reps'][0]['phases']['bottom']=[5,6]
    result=build_metrics(pose,segments)['reps'][0]
    assert result['detect']['bottom'] == .4
    assert result['excluded']


@pytest.mark.parametrize('mutation,reason',[
    ('outside','무릎 관절 화면 밖'),('invisible','무릎 관절 visibility 부족')])
def test_bad_dependency_marks_cell_unavailable_with_a_reason(mutation,reason):
    pose=_pose();segments=plate_segments()
    for frame in range(4,9):
        if mutation=='outside':pose['lms'][frame,25,0]=1.2
        else:pose['lms'][frame,25,3]=.49
    result=build_metrics(pose,segments)
    bottom=result['reps'][0]['positions']['BOTTOM']
    assert bottom['values']['knee_angle'] is None
    assert reason in bottom['reasons']['knee_angle']
    # An unrelated arm metric is not part of the squat display; hip uses knee too.
    assert bottom['values']['hip_angle'] is None
    assert bottom['values']['trunk_lean'] is not None


def test_three_joint_valid_frames_are_enough_and_point_five_is_inclusive():
    pose=_pose();segments=plate_segments()
    pose['lms'][4:9,25,3]=.5
    pose['lms'][4:6,25,3]=.49
    bottom=build_metrics(pose,segments)['reps'][0]['positions']['BOTTOM']
    assert bottom['values']['knee_angle'] is not None
    assert bottom['reasons']['knee_angle'] is None
    pose['lms'][6,25,3]=.49
    assert build_metrics(pose,segments)['reps'][0]['positions']['BOTTOM']['values']['knee_angle'] is None


def test_raw_observations_cannot_be_hidden_by_smoothing():
    pose=_pose();segments=plate_segments()
    pose['raw_lms']=pose['lms'].copy()
    pose['raw_lms'][4:9,25,1]=-0.01
    bottom=build_metrics(pose,segments)['reps'][0]['positions']['BOTTOM']
    assert bottom['values']['knee_angle'] is None
    assert '화면 밖' in bottom['reasons']['knee_angle']


def test_unavailable_reason_survives_findings():
    from feedback.compare import compare_reps
    pose=_pose();segments=plate_segments()
    user=build_metrics(pose,segments)
    ref=deepcopy(user)
    user['reps'][0]['positions']['BOTTOM']['values']['knee_angle']=None
    user['reps'][0]['positions']['BOTTOM']['reasons']['knee_angle']='무릎 관절 화면 밖'
    findings=compare_reps(user,ref,1,allow_excluded_reference=True)
    cell=next(c for c in findings['reps'][0]['cells'] if c['where']=='BOTTOM' and c['metric']=='knee_angle')
    assert cell['status']=='unavailable'
    assert '사용자: 무릎 관절 화면 밖' in cell['unavailable_reason']


def test_ratio_metrics_require_the_normalization_joints_too():
    pose=_pose();segments=plate_segments();segments['exercise']='benchpress'
    pose['lms'][4:9,24,3]=.49
    bottom=build_metrics(pose,segments)['reps'][0]['positions']['BOTTOM']
    assert bottom['values']['elbow_angle'] is not None
    assert bottom['values']['wrist_elbow_x_diff'] is None
    assert '엉덩이 관절 visibility 부족' in bottom['reasons']['wrist_elbow_x_diff']


def test_plate_cache_preserves_raw_landmarks_separately_from_smoothed(monkeypatch,tmp_path):
    from feedback import pose as module
    import export_overlay
    class Sequence(list):pass
    smooth=_pose(total=1)['lms'][0]
    raw=smooth.copy();raw[25,0]=1.1
    sequence=Sequence([smooth]);sequence.raw_lms=[raw]
    monkeypatch.setattr(export_overlay,'video_info',lambda p:{'fps':30,'w':640,'h':480})
    monkeypatch.setattr(module,'extract_backend_sequence',lambda *args:sequence)
    video=tmp_path/'one.mp4';video.touch();cache=tmp_path/'pose.npz'
    first=module.extract_pose(video,cache,'squat',plate_rows=[{'seq':0}])
    second=module.extract_pose(video,cache,'squat',plate_rows=[{'seq':0}])
    assert second['cache_hit']
    assert first['lms'][0,25,0] <= 1
    assert second['raw_lms'][0,25,0] > 1


def test_stage0_distance_population_and_screen_filtered_wrist(tmp_path):
    from tools.s3a_pose_audit import distance_samples
    from core.io.rows import write_rows
    data=np.ones((4,2,33,4),np.float32)
    data[:,:,:,:2]=(.25,.4)
    data[:,:,15,:2]=(.3,.4)
    data[:,:,16,:2]=(.9,.9)
    data[3,:,15,0]=1.1;data[3,:,16,0]=1.2
    np.savez_compressed(tmp_path/'candidates.npz',lms=data,counts=[1,2,1,1],w=400,h=300)
    (tmp_path/'plate').mkdir()
    rows=[{'seq':i,'t':i/30,'plate':{'used':i!=2,'box':[100,100,120,120]}} for i in range(4)]
    write_rows(tmp_path/'plate/rows.jsonl.gz',rows)
    samples=distance_samples(tmp_path)
    assert [s['seq'] for s in samples]==[0,3]
    assert samples[0]['wrist']==pytest.approx(np.hypot(10,10)/20)
    assert samples[1]['wrist'] is None
    assert samples[1]['ranking']==samples[1]['box']
