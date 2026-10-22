"""Synthetic checks for the report 3D tab runtime."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import pytest


def _template() -> str:
    return Path("web/report_template.html").read_text(encoding="utf-8")


def _template_script() -> str:
    match = re.search(r"<script>\s*/\* Video, sprite.*?</script>", _template(), flags=re.S)
    assert match is not None
    return re.sub(r"^<script>\s*|\s*</script>$", "", match.group(0), flags=re.S)


def _run_report3d(body: str) -> dict:
    if subprocess.run(["node", "--version"], capture_output=True).returncode != 0:
        pytest.skip("node is not available")
    script = """
const vm = require("vm");
const context = {console};
vm.createContext(context);
vm.runInContext(%s, context);
const R = context.Report3D;
%s
""" % (json.dumps(_template_script()), body)
    completed = subprocess.run(["node"], input=script, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_template_loads_classic_three_and_pose3d_scripts_before_main_code():
    html = _template()

    assert '<script src="vendor/three.min.js"></script>' in html
    assert "<!--__POSE3D_SCRIPTS__-->" in html
    assert html.index('src="vendor/three.min.js"') < html.index("const D =")
    assert html.index("<!--__POSE3D_SCRIPTS__-->") < html.index("const D =")
    assert 'type="module"' not in html
    assert "fetch(" not in html
    assert "H36M_METRIC_JOINTS" not in html
    assert "POSE3D_META" not in html


def test_report3d_display_frame_converts_axes_mirrors_ref_and_sets_floor_to_lower_ankle():
    result = _run_report3d(
        """
const frame = Array.from({length:17}, (_, i) => [i, i / 10, i / 100]);
frame[3] = [3, 1.2, .3];
frame[6] = [6, 1.8, .6];
const data = {joints:"h36m17", total:2, frames:[frame, null]};
const converted = R.displayFrame(data, 0, {mirror:true});
console.log(JSON.stringify({
  valid: R.validate(data, 2),
  badTotal: R.validate(data, 3),
  pelvis: converted[0],
  rightAnkle: converted[3],
  leftAnkle: converted[6],
  missing: R.displayFrame(data, 1) === null
}));
"""
    )

    assert result["valid"] is True
    assert result["badTotal"] is False
    assert result["pelvis"] == [0, 1.8, 0]
    assert result["rightAnkle"] == [-3, 0.6000000000000001, -0.3]
    assert result["leftAnkle"] == [-6, 0, -0.6]
    assert result["missing"] is True


def test_report3d_goal_joints_use_h36m_metric_mapping_and_role_side():
    result = _run_report3d(
        """
const report = {meta:{pose3d:{metric_joints:{
  knee_angle:{right:[1,2,3],left:[4,5,6]},
  hip_angle:{right:[14,1,2],left:[11,4,5]},
  trunk_lean:{right:[0,7,8],left:[0,7,8]},
  foot_flatness:{right:[3],left:[6]},
  head_hip_line:{right:[10,0],left:[10,0]}
}}}, user:{total:100}, ref:{total:80}};
const finding = {goal:{type:"group", metrics:["knee_angle","hip_angle"], metric:"knee_angle"}};
console.log(JSON.stringify({
  user: R.goalJoints(report, finding, "user", {side:"right"}, {side:"left"}),
  ref: R.goalJoints(report, finding, "ref", {side:"right"}, {side:"left"}),
  frames: R.frameIndices(report, 120, -5),
  top: R.cameraPreset("top")
}));
"""
    )

    assert result["user"] == [1, 2, 3, 14]
    assert result["ref"] == [4, 5, 6, 11]
    assert result["frames"] == {"user": 99, "ref": 0}
    assert result["top"][1] > result["top"][0]


def test_report3d_missing_metadata_uses_empty_mapping_without_static_constants():
    result = _run_report3d(
        """
const report = {meta:{pose3d:{}}, user:{total:10}, ref:{total:10}};
const finding = {goal:{type:"single", metric:"knee_angle"}};
console.log(JSON.stringify({
  joints: R.goalJoints(report, finding, "user", {side:"right"}, {side:"left"}),
  frames: R.frameIndices(report, 1, 2)
}));
"""
    )

    assert result == {"joints": [], "frames": {"user": 1, "ref": 2}}


def test_report3d_camera_fit_uses_display_extents_with_projection_margin():
    result = _run_report3d(
        """
const tall = [[-.3,0,0],[.3,3.2,0],[0,1.6,.2],[0,1.6,-.2]];
const fit = R.fitCamera(tall, {aspect:1, fov:45, preset:"side"});
const halfHeightAtTarget = Math.tan(45 * Math.PI / 360) * fit.distance;
console.log(JSON.stringify({
  targetY: fit.target[1],
  distance: fit.distance,
  halfHeightAtTarget,
  requiredHalfHeight: fit.bounds.size[1] / 2
}));
"""
    )

    assert result["targetY"] == pytest.approx(1.6)
    assert result["distance"] > 4.0
    assert result["halfHeightAtTarget"] > result["requiredHalfHeight"] * 1.4


def test_report3d_fit_bounds_match_actual_group_world_offsets_once():
    result = _run_report3d(
        """
const user = [[0,0,0],[.2,1,0],[.4,2,0]];
const ref = [[0,0,0],[-.2,1,0],[-.4,2,0]];
const userLocal = R.scenePoints(user);
const refLocal = R.scenePoints(ref);
const fitPoints = [...R.offsetPoints(userLocal, -.55), ...R.offsetPoints(refLocal, .55)];
const actualWorld = [
  ...userLocal.map(p => [p[0] - .55, p[1], p[2]]),
  ...refLocal.map(p => [p[0] + .55, p[1], p[2]])
];
console.log(JSON.stringify({fit: R.bounds(fitPoints), actual: R.bounds(actualWorld)}));
"""
    )

    assert result["fit"] == result["actual"]


def test_report3d_back_fit_uses_front_axes_not_side_axes():
    result = _run_report3d(
        """
const points = [[-2,0,-.1],[2,1,.1],[0,2,.2]];
const front = R.fitCamera(points, {aspect:1, fov:45, preset:"front", margin:1});
const back = R.fitCamera(points, {aspect:1, fov:45, preset:"back", margin:1});
const side = R.fitCamera(points, {aspect:1, fov:45, preset:"side", margin:1});
console.log(JSON.stringify({front: front.distance, back: back.distance, side: side.distance}));
"""
    )

    assert result["back"] == pytest.approx(result["front"])
    assert result["back"] > result["side"]


def test_report3d_camera_preset_recomputes_fit_for_that_preset():
    if subprocess.run(["node", "--version"], capture_output=True).returncode != 0:
        pytest.skip("node is not available")
    source = _template_script()
    pieces = [
        r"function pose3dBounds\(points\)\{.*?\n\}",
        r"function pose3dFitCamera\(points,\{aspect=1,fov=45,preset=\"side\",margin=1\.45\}=\{\}\)\{.*?\n\}",
        r"function pose3dApplyCamera\(\)\{.*?\n\}",
        r"function pose3dSetCamera\(name\)\{.*?\n\}",
    ]
    extracted = []
    for pattern in pieces:
        match = re.search(pattern, source, flags=re.S)
        assert match is not None
        extracted.append(match.group(0))
    script = """
let calls = 0;
const clamp = (v,a,b) => Math.max(a,Math.min(b,v));
function pose3dCameraPreset(name,distance=3){
  const presets={side:[distance,1.35,0],front:[0,1.35,distance],top:[0,distance+.8,.01],back:[0,1.35,-distance]};
  return presets[name]||presets.side;
}
let pose3dView = {
  preset: "side",
  distance: 3,
  scenePoints: [[-2,0,-.1],[2,1,.1],[0,2,.2]],
  fit: {target:[0,1,0], distance: 2.2},
  target: [0,1,0],
  drag: {distance: 2.2, elevation: 0, azimuth: 0},
  camera: {
    aspect: 1,
    fov: 45,
    position: {x:0,y:0,z:0,set(x,y,z){this.x=x;this.y=y;this.z=z;}},
    lookAt(){},
  },
  renderer: {render(){calls += 1;}},
  scene: {},
};
%s
const side = pose3dFitCamera(pose3dView.scenePoints, {aspect:1, fov:45, preset:"side"}).distance;
const back = pose3dFitCamera(pose3dView.scenePoints, {aspect:1, fov:45, preset:"back"}).distance;
pose3dSetCamera("back");
console.log(JSON.stringify({calls, side, back, used: pose3dView.fit.distance, preset: pose3dView.preset}));
""" % "\n".join(extracted)
    completed = subprocess.run(["node"], input=script, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)

    assert result["calls"] == 1
    assert result["preset"] == "back"
    assert result["back"] > result["side"]
    assert result["used"] == pytest.approx(result["back"])


def test_3d_dom_contains_required_controls_and_placeholder():
    html = _template()

    assert 'id="view-3d"' in html
    assert 'id="pose3d-style"' in html
    for preset in ("side", "front", "top", "back"):
        assert f'data-3d-camera="{preset}"' in html
    assert 'id="pose3d-pip-video"' in html
    for key in ("robot", "side", "front", "top", "back", "no_data", "pip"):
        assert f'p3Label("{key}")' in html


def test_report3d_camera_actions_render_immediately_without_frame_advance():
    if subprocess.run(["node", "--version"], capture_output=True).returncode != 0:
        pytest.skip("node is not available")
    camera_functions = re.search(
        r"function pose3dApplyCamera\(\)\{.*?\n\}\nfunction pose3dSetCamera\(name\)\{.*?\n\}\nfunction pose3dAimCamera\(\)\{.*?\n\}",
        _template_script(),
        flags=re.S,
    )
    assert camera_functions is not None
    script = """
let calls = 0;
let pose3dView = {
  distance: 3,
  drag: {azimuth: 0, elevation: .2, distance: 3},
  camera: {
    position: {x:0,y:0,z:0,set(x,y,z){this.x=x;this.y=y;this.z=z;}},
    lookAt(){},
  },
  renderer: {render(){calls += 1;}},
  scene: {},
  target: [0, 1.6, 0],
  fit: {distance: 5.2},
};
const clamp = (v,a,b) => Math.max(a,Math.min(b,v));
function pose3dCameraPreset(name,distance=3){
  const presets={side:[distance,1.35,0],front:[0,1.35,distance],top:[0,distance+.8,.01],back:[0,1.35,-distance]};
  return presets[name]||presets.side;
}
%s
pose3dSetCamera("front");
const topBeforeDrag = (() => { pose3dSetCamera("top"); return pose3dView.drag.elevation; })();
pose3dView.drag.azimuth += .3;
pose3dAimCamera();
console.log(JSON.stringify({calls, topBeforeDrag, camera:[pose3dView.camera.position.x, pose3dView.camera.position.y, pose3dView.camera.position.z]}));
""" % camera_functions.group(0)
    completed = subprocess.run(["node"], input=script, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)

    assert result["calls"] == 3
    assert 1.5 < result["topBeforeDrag"] < 1.56
    assert result["camera"] != [0, 0, 0]
