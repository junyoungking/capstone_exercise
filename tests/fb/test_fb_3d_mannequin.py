"""Synthetic checks for the 3D mannequin helper contract."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import pytest


PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]


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
const parents = %s;
%s
""" % (json.dumps(_template_script()), json.dumps(PARENTS), body)
    completed = subprocess.run(["node"], input=script, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _three_script() -> str:
    return Path("web/vendor/three.min.js").read_text(encoding="utf-8")


def _run_report3d_runtime(body: str) -> dict:
    if subprocess.run(["node", "--version"], capture_output=True).returncode != 0:
        pytest.skip("node is not available")
    script = """
const vm = require("vm");
const context = {console};
vm.createContext(context);
vm.runInContext(%s, context);
vm.runInContext(%s, context);
vm.runInContext(`
THREE.WebGLRenderer = class {
  constructor(){ this.domElement = {tagName:"CANVAS"}; this.renderCount = 0; }
  setPixelRatio(value){ this.pixelRatio = value; }
  setSize(width, height, updateStyle){ this.size = [width, height, updateStyle]; }
  render(){ this.renderCount += 1; }
};
function makeFrame(armExtent=0){
  const pts = Array.from({length:17}, () => [0,0,0]);
  pts[0]=[0,0,0]; pts[1]=[.18,.85,0]; pts[2]=[.18,1.35,0]; pts[3]=[.18,1.9,0];
  pts[4]=[-.18,.85,0]; pts[5]=[-.18,1.35,0]; pts[6]=[-.18,1.9,0];
  pts[7]=[0,-.24,0]; pts[8]=[0,-.52,0]; pts[9]=[0,-.68,0]; pts[10]=[0,-.82,0];
  pts[11]=[-.32,-.5,0]; pts[12]=[-.62,-.35,0]; pts[13]=[-.92,-.22,0];
  pts[14]=[.32,-.5,0]; pts[15]=[.62 + armExtent / 2,-.35,0]; pts[16]=[.92 + armExtent,-.22,0];
  return pts;
}
function makePose(){
  return {joints:"h36m17", total:5, frames:[makeFrame(0), makeFrame(.25), makeFrame(-.2), makeFrame(.1), makeFrame(0)]};
}
function makeElement(id){
  return {
    id, children: [], style: {}, dataset: {}, attributes: {}, hidden: false, textContent: "",
    clientWidth: 800, clientHeight: 500, offsetTop: 0, offsetHeight: 40, offsetWidth: 60,
    classList: {items:new Set(), toggle(name){const has=this.items.has(name); if(has)this.items.delete(name); else this.items.add(name); return !has;}},
    appendChild(child){ this.children.push(child); child.parentElement = this; },
    setAttribute(name, value){ this.attributes[name] = String(value); },
    getAttribute(name){ return this.attributes[name]; },
    setPointerCapture(){}, releasePointerCapture(){},
    querySelector(selector){ return this.selectors?.[selector] || null; },
    querySelectorAll(selector){ return this.selectorLists?.[selector] || []; },
  };
}
function setupRuntime(){
  const root=makeElement("view-3d"), toolbar=makeElement("toolbar"), canvas=makeElement("pose3d-canvas");
  const message=makeElement("pose3d-message"), pip=makeElement("pose3d-pip"), pipVideo=makeElement("pose3d-pip-video");
  const pipToggle=makeElement("pose3d-pip-toggle"), userLabel=makeElement("user-label"), refLabel=makeElement("ref-label");
  toolbar.offsetTop=12; toolbar.offsetHeight=32; pip.offsetTop=12; pip.offsetHeight=88;
  const cameraButtons=["side","front","top","back"].map(name => ({dataset:{"3dCamera":name}, attributes:{}, setAttribute(k,v){this.attributes[k]=String(v);}}));
  const styleButtons=["robot","stick"].map(name => ({dataset:{"3dStyle":name}, attributes:{}, setAttribute(k,v){this.attributes[k]=String(v);}}));
  root.selectors={".pose3d-toolbar": toolbar, '[data-3d-role-label="user"]': userLabel, '[data-3d-role-label="ref"]': refLabel};
  root.selectorLists={"[data-3d-camera]": cameraButtons, "[data-3d-style]": styleButtons};
  const ids={"view-3d":root,"pose3d-message":message,"pose3d-canvas":canvas,"pose3d-pip":pip,"pose3d-pip-video":pipVideo,"pose3d-pip-toggle":pipToggle};
  document={getElementById(id){return ids[id] || makeElement(id);}, querySelectorAll(){return [];}, addEventListener(){}, createElement:makeElement};
  window={devicePixelRatio:1};
  ResizeObserver=class { constructor(callback){ this.callback=callback; } observe(element){ this.element=element; } };
  W={
    user:{total:5,fps:30,reps:[{rep:1,side:"right",positions:{START:0,MID_DOWN:1,BOTTOM:2,MID_UP:3,LOCKOUT:3},phases:{down:[0,2],bottom:[2,3],up:[3,4]}},
                            {rep:2,side:"right",positions:{START:1,MID_DOWN:2,BOTTOM:3,MID_UP:3,LOCKOUT:4},phases:{down:[1,3],bottom:[3,4],up:[4,5]}}]},
    ref:{total:5,fps:30,mirror:false,reps:[{rep:1,side:"right",positions:{START:0,MID_DOWN:1,BOTTOM:2,MID_UP:3,LOCKOUT:3},phases:{down:[0,2],bottom:[2,3],up:[3,4]}}]},
    reps:[{rep:1,goal:{type:"none"}},{rep:2,goal:{type:"none"}}],
    meta:{positions:["START","MID_DOWN","BOTTOM","MID_UP","LOCKOUT"],
      pose3d:{parents:%s,metric_joints:{},labels:{no_data:"3D 데이터 없음"},roles:{user:"사용자",ref:"레퍼런스"}}}
  };
  globalThis.POSE3D={user:makePose(), ref:makePose()};
  uv={src:"user.mp4",currentTime:0,paused:true};
  view="3d"; curRep=0; curFrame=1; refFrame=1; total=5;
  return {root,toolbar,canvas,message,pip,pipVideo,pipToggle,userLabel,refLabel,cameraButtons,styleButtons};
}
`, context);
vm.runInContext(%s, context);
""" % (
        json.dumps(_three_script()),
        json.dumps(_template_script()),
        json.dumps(PARENTS),
        json.dumps("const parents = %s;\n%s" % (json.dumps(PARENTS), body)),
    )
    completed = subprocess.run(["node"], input=script, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_bone_profile_uses_video_wide_median_lengths_and_skips_invalid_frames():
    result = _run_report3d(
        """
function frame(length1, torso){
  const pts = Array.from({length:17}, () => [0,0,0]);
  pts[1] = [length1, 0, 0];
  pts[7] = [0, torso / 2, 0];
  pts[8] = [0, torso, 0];
  return pts;
}
const data = {
  joints: "h36m17",
  total: 5,
  frames: [
    frame(1, .4),
    null,
    frame(100, .5),
    frame(3, .6),
    frame(3, .6).map((p, i) => i === 1 ? [NaN, 0, 0] : p),
  ],
};
const profile = R.boneProfile(data, parents);
console.log(JSON.stringify({
  root: profile.lengths[0],
  hipChild: profile.lengths[1],
  torsoLength: profile.torsoLength,
  lengthCount: profile.lengths.length,
}));
"""
    )

    assert result["lengthCount"] == 17
    assert result["root"] == 0
    assert result["hipChild"] == pytest.approx(3)
    assert result["torsoLength"] == pytest.approx(0.5)


def test_fixed_bones_preserves_root_anchor_direction_lengths_and_input_points():
    result = _run_report3d(
        """
const points = Array.from({length:17}, () => [0,0,0]);
points[0] = [10, 20, 30];
points[1] = [13, 24, 30];
points[2] = [15, 24, 30];
points[4] = null;
const before = JSON.stringify(points);
const profile = {lengths: Array.from({length:17}, () => 1), torsoLength: 1};
profile.lengths[1] = 10;
profile.lengths[2] = 5;
const fixed = R.fixedBones(points, profile, parents);
console.log(JSON.stringify({
  root: fixed[0],
  child: fixed[1],
  grandchild: fixed[2],
  nullChild: fixed[4],
  unchanged: JSON.stringify(points) === before,
}));
"""
    )

    assert result["root"] == [10, 20, 30]
    assert result["child"] == pytest.approx([16, 28, 30])
    assert result["grandchild"] == pytest.approx([21, 28, 30])
    assert result["nullChild"] is None
    assert result["unchanged"] is True


def test_robot_frame_mirrors_converts_to_display_space_fixes_lengths_and_sets_floor():
    result = _run_report3d(
        """
function frame(rightAnkleY, leftAnkleY, childX){
  const pts = Array.from({length:17}, () => [0,0,0]);
  pts[1] = [childX, 0, 0];
  pts[3] = [0, rightAnkleY, 0];
  pts[6] = [0, leftAnkleY, 0];
  pts[7] = [0, .25, 0];
  pts[8] = [0, .5, 0];
  return pts;
}
const data = {joints:"h36m17", total:3, frames:[frame(1, 2, 1), frame(1, 2, 100), frame(1, 2, 3)]};
const robot = R.robotFrame(data, 1, {mirror:true}, parents);
console.log(JSON.stringify({
  pelvis: robot[0],
  child: robot[1],
  rightAnkle: robot[3],
  leftAnkle: robot[6],
}));
"""
    )

    assert result["pelvis"] == [0, 2, 0]
    assert result["child"] == pytest.approx([-3, 2, 0])
    assert result["leftAnkle"][1] == pytest.approx(0)
    assert result["rightAnkle"][1] == pytest.approx(1)


def test_robot_frame_returns_null_when_selected_frame_has_no_pose():
    result = _run_report3d(
        """
const data = {joints:"h36m17", total:2, frames:[Array.from({length:17}, () => [0,0,0]), null]};
console.log(JSON.stringify({missing: R.robotFrame(data, 1, {}, parents)}));
"""
    )

    assert result["missing"] is None


def test_mannequin_spec_returns_scaled_body_part_radii_from_torso_length():
    result = _run_report3d(
        """
const small = R.mannequinSpec({lengths:Array(17).fill(1), torsoLength:.25});
const base = R.mannequinSpec({lengths:Array(17).fill(1), torsoLength:.5});
console.log(JSON.stringify({
  base,
  smallHead: small.headRadius,
  baseHead: base.headRadius,
  order: [
    base.torsoRadius > base.thighRadius,
    base.thighRadius > base.shinRadius,
    base.shinRadius > base.upperArmRadius,
    base.upperArmRadius > base.forearmRadius,
    base.jointRadius > base.forearmRadius,
  ],
}));
"""
    )

    assert result["base"]["scale"] == pytest.approx(1.55)
    assert result["base"]["headRadius"] == pytest.approx(0.10 * 1.55, abs=0.01)
    assert result["base"]["torsoRadius"] == pytest.approx(0.13 * 1.55, abs=0.01)
    assert result["base"]["thighRadius"] == pytest.approx(0.075 * 1.55, abs=0.01)
    assert result["base"]["shinRadius"] == pytest.approx(0.055 * 1.55, abs=0.01)
    assert result["base"]["upperArmRadius"] == pytest.approx(0.045 * 1.55, abs=0.01)
    assert result["base"]["forearmRadius"] == pytest.approx(0.04 * 1.55, abs=0.01)
    assert result["smallHead"] == pytest.approx(result["baseHead"] / 2)
    assert all(result["order"])


def test_rep_scene_points_include_user_and_ref_rep_extremes_with_offsets_once():
    result = _run_report3d(
        """
function frame(armExtent, ankleY){
  const pts = Array.from({length:17}, () => [0,0,0]);
  pts[1] = [.2, 0, 0];
  pts[3] = [0, ankleY, 0];
  pts[6] = [0, ankleY + .2, 0];
  pts[7] = [0, .25, 0];
  pts[8] = [0, .5, 0];
  pts[14] = [Math.sign(armExtent || 1) * .25, .5, 0];
  pts[15] = [armExtent / 2, .5, 0];
  pts[16] = [armExtent, .5, 0];
  return pts;
}
const report = {user:{total:5}, ref:{total:5}, meta:{pose3d:{parents}}};
const poses = {
  user: {joints:"h36m17", total:5, frames:[frame(99,1), frame(1.5,1), frame(-1.5,1), frame(0,1), frame(-99,1)]},
  ref: {joints:"h36m17", total:5, frames:[frame(-99,1), frame(-1.5,1), frame(1.5,1), frame(0,1), frame(99,1)]},
};
const before = JSON.stringify(poses);
const points = R.repScenePoints(report, poses, {positions:{START:1, LOCKOUT:3}}, {positions:{START:1, LOCKOUT:3}});
const xs = points.map(p => p[0]).sort((a,b) => a-b);
console.log(JSON.stringify({
  minX: xs[0],
  maxX: xs[xs.length - 1],
  hasMiddleUserMin: points.some(p => p[0] < -2.8 && p[0] > -3.0),
  hasMiddleRefMax: points.some(p => p[0] > 2.8 && p[0] < 3.0),
  count: points.length,
  unchanged: JSON.stringify(poses) === before,
}));
"""
    )

    assert result["count"] > 0
    assert result["minX"] > -4.0
    assert result["maxX"] > 2.8
    assert result["maxX"] < 4.0
    assert result["hasMiddleUserMin"] is True
    assert result["hasMiddleRefMax"] is True
    assert result["unchanged"] is True


def test_3d_dom_exposes_segmented_style_role_labels_and_collapsible_pip():
    html = _template()

    assert 'id="pose3d-style" role="group"' in html
    assert 'data-3d-style="robot" aria-pressed="true"' in html
    assert 'data-3d-style="stick" aria-pressed="false"' in html
    assert 'data-3d-role-label="user"' in html
    assert 'data-3d-role-label="ref"' in html
    assert 'id="pose3d-pip-toggle" aria-expanded="true"' in html
    assert ".pose3d-pip{position:absolute;right:12px;top:12px" in html
    assert ".pose3d-pip.collapsed video{display:none}" in html


def test_runtime_robot_style_builds_mannequin_parts_with_scaled_capsules():
    result = _run_report3d_runtime(
        """
setupRuntime();
const role = pose3dRoleObjects("user", -.55);
const spec = Report3D.mannequinSpec({lengths:Array(17).fill(.5), torsoLength:.5});
const points = Report3D.scenePoints(makeFrame(.2));
pose3dView = {style:"robot"};
role.spec = spec;
updatePose3dRole(role, points, [16]);
const headMid = points[9].map((v,i)=>(v+points[10][i])/2);
const bone = index => role.bones.find(b => b && b.index === index);
console.log(JSON.stringify({
  head: role.head.position.toArray(),
  headRadius: role.head.scale.x,
  headVisible: role.head.visible,
  torsoRadii: [bone(7).radius, bone(8).radius],
  limbRadii: [bone(2).radius, bone(3).radius, bone(12).radius, bone(13).radius],
  connectorCount: role.connectors.length,
  connectorRadii: role.connectors.map(c => c.radius),
  connectorVisible: role.connectors.every(c => c.mesh.visible && c.caps.every(cap => cap.visible)),
  matte: {type: bone(8).mesh.material.type, roughness: bone(8).mesh.material.roughness, metalness: bone(8).mesh.material.metalness},
  highlight: role.highlights[16].visible,
  stickHidden: role.stickLine.visible,
  headMid,
}));
"""
    )

    assert result["head"] == pytest.approx(result["headMid"])
    assert result["headRadius"] == pytest.approx(0.10 * 1.55, abs=0.01)
    assert result["headVisible"] is True
    assert result["torsoRadii"] == pytest.approx([0.13 * 1.55, 0.13 * 1.55], abs=0.01)
    assert result["limbRadii"] == pytest.approx([0.075 * 1.55, 0.055 * 1.55, 0.045 * 1.55, 0.04 * 1.55], abs=0.01)
    assert result["connectorCount"] == 2
    assert result["connectorRadii"] == pytest.approx([0.13 * 1.55, 0.13 * 1.55], abs=0.01)
    assert result["connectorVisible"] is True
    assert result["matte"] == {"type": "MeshStandardMaterial", "roughness": pytest.approx(0.88), "metalness": 0}
    assert result["highlight"] is True
    assert result["stickHidden"] is False


def test_runtime_stick_style_keeps_line_segments_and_hides_mannequin_capsules():
    result = _run_report3d_runtime(
        """
setupRuntime();
const role = pose3dRoleObjects("ref", .55);
const points = Report3D.scenePoints(makeFrame(-.15));
pose3dView = {style:"stick"};
role.spec = Report3D.mannequinSpec({lengths:Array(17).fill(.5), torsoLength:.5});
updatePose3dRole(role, points, []);
const firstEdge = role.edges[0];
const line = Array.from(role.stickLine.geometry.attributes.position.array.slice(0, 6));
console.log(JSON.stringify({
  stickVisible: role.stickLine.visible,
  lineMaterial: role.stickLine.material.type,
  firstEdge,
  line,
  expected: [...points[firstEdge[0]], ...points[firstEdge[1]]],
  capsulesHidden: role.bones.filter(Boolean).every(b => !b.mesh.visible && b.caps.every(cap => !cap.visible)),
  headHidden: role.head.visible,
  jointScale: role.joints[0].scale.x,
  jointMaterialIsStick: role.joints[0].material === role.stickJointMat,
}));
"""
    )

    assert result["stickVisible"] is True
    assert result["lineMaterial"] == "LineBasicMaterial"
    assert result["line"] == pytest.approx(result["expected"])
    assert result["capsulesHidden"] is True
    assert result["headHidden"] is False
    assert result["jointScale"] == pytest.approx(0.62)
    assert result["jointMaterialIsStick"] is True


def test_runtime_camera_fit_stays_fixed_for_frame_and_style_and_refits_for_resize_or_rep_change():
    result = _run_report3d_runtime(
        """
const dom = setupRuntime();
updatePose3d();
const first = {debug: Report3D.debug(), quat: pose3dView.camera.quaternion.toArray()};
curFrame = 2; refFrame = 2; updatePose3d();
const frame = {debug: Report3D.debug(), quat: pose3dView.camera.quaternion.toArray()};
pose3dView.style = "stick"; updatePose3d();
const style = {debug: Report3D.debug(), quat: pose3dView.camera.quaternion.toArray()};
dom.root.clientWidth = 920; updatePose3d();
const resize = {debug: Report3D.debug(), quat: pose3dView.camera.quaternion.toArray()};
curRep = 1; curFrame = 2; refFrame = 2; updatePose3d();
const rep = {debug: Report3D.debug(), quat: pose3dView.camera.quaternion.toArray()};
console.log(JSON.stringify({
  firstCount: first.debug.fitCount,
  frameCount: frame.debug.fitCount,
  styleCount: style.debug.fitCount,
  resizeCount: resize.debug.fitCount,
  repCount: rep.debug.fitCount,
  frameSameCamera: JSON.stringify(frame.debug.camera) === JSON.stringify(first.debug.camera),
  styleSameCamera: JSON.stringify(style.debug.camera) === JSON.stringify(first.debug.camera),
  frameSameQuat: JSON.stringify(frame.quat) === JSON.stringify(first.quat),
  styleSameQuat: JSON.stringify(style.quat) === JSON.stringify(first.quat),
  resizeChangedKey: resize.debug.fitKey !== first.debug.fitKey,
  repChangedKey: rep.debug.fitKey !== resize.debug.fitKey,
  rendererSize: pose3dView.renderer.size,
}));
"""
    )

    assert result["firstCount"] == 1
    assert result["frameCount"] == result["firstCount"]
    assert result["styleCount"] == result["firstCount"]
    assert result["resizeCount"] == result["firstCount"] + 1
    assert result["repCount"] == result["resizeCount"] + 1
    assert result["frameSameCamera"] is True
    assert result["styleSameCamera"] is True
    assert result["frameSameQuat"] is True
    assert result["styleSameQuat"] is True
    assert result["resizeChangedKey"] is True
    assert result["repChangedKey"] is True
    assert result["rendererSize"][:2] == [920, 500]


def test_runtime_goal_marker_stays_visible_over_robot_and_restores_stick_material():
    result = _run_report3d_runtime(
        """
setupRuntime();
const role = pose3dRoleObjects("user", -.55);
const points = Report3D.scenePoints(makeFrame(.2));
role.spec = Report3D.mannequinSpec({lengths:Array(17).fill(.5), torsoLength:.5});
pose3dView = {style:"robot"};
updatePose3dRole(role, points, [2]);
const marker = role.highlights[2];
const robot = {visible:marker.visible, position:marker.position.toArray(),
  depthTest:marker.material.depthTest, depthWrite:marker.material.depthWrite,
  renderOrder:marker.renderOrder, color:marker.material.color.getHex(),
  radius:marker.geometry.parameters.radius * marker.scale.x};
pose3dView.style = "stick";
updatePose3dRole(role, points, [2]);
console.log(JSON.stringify({robot, expected:points[2], bodyOrder:role.bones[2].mesh.renderOrder,
  stick:{depthTest:marker.material.depthTest, depthWrite:marker.material.depthWrite,
    renderOrder:marker.renderOrder, radius:marker.geometry.parameters.radius * marker.scale.x}}));
"""
    )

    assert result["robot"]["visible"] is True
    assert result["robot"]["position"] == pytest.approx(result["expected"])
    assert result["robot"]["depthTest"] is False
    assert result["robot"]["depthWrite"] is False
    assert result["robot"]["renderOrder"] > result["bodyOrder"]
    assert result["robot"]["color"] == 0xD93F3F
    assert result["robot"]["radius"] == pytest.approx(0.05 * 1.55)
    assert result["stick"] == {"depthTest": True, "depthWrite": True, "renderOrder": 0, "radius": 0.06}
