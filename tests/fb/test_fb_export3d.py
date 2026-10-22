"""Local 3D resources are optional and never change comparison values."""
from copy import deepcopy
import json
import subprocess

from feedback import config
from feedback.export import export_report
from test_fb_export import _pose, _segment, _measured, _findings, fake_sprite


def _export(tmp_path, findings):
    video = tmp_path / 'synthetic.mp4'
    video.touch()
    results = {role: {'pose': _pose(), 'segments': _segment(video)} for role in ('user', 'ref')}
    measured = {role: _measured() for role in results}
    return export_report(results, measured, findings, tmp_path)


def test_3d_export_injects_existing_local_scripts_and_copies_classic_vendor(tmp_path, fake_sprite):
    (tmp_path / 'pose3d_user.js').write_text('window.POSE3D={user:{}};', encoding='utf-8')
    findings = _findings()
    report = _export(tmp_path, findings)
    html = (tmp_path / 'report.html').read_text(encoding='utf-8')
    assert '<script src="pose3d_user.js"></script>' in html
    assert '<script src="pose3d_ref.js"></script>' not in html
    assert '<script src="vendor/three.min.js"></script>' in html
    assert '<!--__POSE3D_SCRIPTS__-->' not in html
    for name in ('three.min.js', 'three.LICENSE.txt', 'three.version.json'):
        assert (tmp_path / 'vendor' / name).read_bytes() == (config.PROJECT_ROOT / 'web/vendor' / name).read_bytes()
    assert report['findings'] == findings
    assert report['meta']['pose3d']['parents'] == list(config.POSE3D_PARENTS)
    assert report['meta']['pose3d']['roles'] == config.ROLE_LABELS
    assert report['meta']['pose3d']['metric_joints']['hip_angle']['right'] == [14, 1, 2]


def test_3d_export_without_pose_files_has_no_missing_file_requests(tmp_path, fake_sprite):
    report = _export(tmp_path, _findings())
    html = (tmp_path / 'report.html').read_text(encoding='utf-8')
    assert '<script src="pose3d_user.js"></script>' not in html
    assert '<script src="pose3d_ref.js"></script>' not in html
    assert 'fetch(' not in html and 'type="module"' not in html
    assert report['pose3d'] == {'user': 'pose3d_user.js', 'ref': 'pose3d_ref.js'}


def test_changing_3d_file_leaves_findings_and_panel_values_unchanged(tmp_path, fake_sprite):
    findings = _findings()
    before = _export(tmp_path, deepcopy(findings))
    for role in ('user', 'ref'):
        (tmp_path / f'pose3d_{role}.js').write_text(
            f'window.POSE3D=window.POSE3D||{{}};window.POSE3D.{role}=' +
            json.dumps({'joints': 'h36m17', 'fps': 30, 'total': 40,
                        'frames': [[[999, -999, 123]] * 17] * 40}) + ';', encoding='utf-8')
    after = _export(tmp_path, findings)
    assert after['findings'] == before['findings'] == findings
    assert after['reps'] == before['reps']
    assert after['user']['frame_metrics'] == before['user']['frame_metrics']


def test_vendor_is_requested_npm_classic_three_version():
    source = config.PROJECT_ROOT / 'web/vendor/three.min.js'
    script = "const vm=require('vm'),fs=require('fs'),ctx={};vm.createContext(ctx);" \
             "vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),ctx);" \
             "process.stdout.write(ctx.THREE.REVISION);"
    completed = subprocess.run(['node', '-e', script, str(source)], text=True, capture_output=True, check=True)
    assert completed.stdout == '147'
    metadata = json.loads(source.with_name('three.version.json').read_text(encoding='utf-8'))
    assert metadata['version'] == '0.147.0'
    assert metadata['tarball_url'].startswith('https://registry.npmjs.org/three/')
