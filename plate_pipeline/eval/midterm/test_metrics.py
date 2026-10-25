from pathlib import Path
import numpy as np
import pandas as pd
from reproduce import aggregate


def test_original_summary_formulas_keep_micro_event_f1_and_macro_count(tmp_path):
    # The real formulas are loaded from a minimal source with the same contract.
    # This checks evaluation wiring: per-video mean vs pooled event counts.
    source=Path('/home/kojy1009/capstone_exercise/new_data/experiments_athlete_pipeline_v40/athlete_yolo_balanced_phase_sameconditions_seed42')
    if not source.exists():
        import pytest
        pytest.skip('local v40 source is not part of the repository')
    d=pd.DataFrame([dict(gt_count=2,pred_count=1,abs_error=1,exact=False,obo=True,tp=1,fp=0,fn=1),
                    dict(gt_count=1,pred_count=3,abs_error=2,exact=False,obo=False,tp=1,fp=2,fn=0)])
    s=aggregate(source,d,[-.2,.4])
    assert s['videos']==2 and s['count_mae']==1.5 and s['count_exact']==0 and s['obo']==.5
    assert (s['event_tp'],s['event_fp'],s['event_fn'])==(2,2,1)
    assert s['event_f1']==4/7
    assert np.isclose(s['matched_delay_median_s'],.1)
    assert np.isclose(s['matched_abs_delay_p95_s'],.39)


def test_fixed_no_plate_exclusion_does_not_expand_for_failed_detections():
    from cohort import included,HERE
    exclusions=pd.read_csv(HERE/'excluded_no_plate.csv')
    other=[f'squat/eligible_{i}.mp4' for i in range(61)]
    frame=pd.DataFrame({'key':exclusions.key.tolist()+other})
    frame['detected']=False
    result=included(frame)
    assert len(result)==61 and set(result.key)==set(other)
    assert not result.detected.any()  # missing detection never enlarges exclusions
