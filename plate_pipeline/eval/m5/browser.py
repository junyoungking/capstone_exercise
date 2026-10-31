"""Real local auto -> confirmation -> saved report flow, both viewport sizes."""
from pathlib import Path
import json,time
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'eval_out/m5/browser';out.mkdir(parents=True,exist_ok=True)
video='/home/kojy1009/work/mc-count/plate_pipeline/eval_out/mc/inputs/user_clip_1.mp4';url='http://127.0.0.1:8012'
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader']);page=browser.new_page(viewport={'width':1920,'height':1080});errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
 def shot(name):
  for w,h in ((1920,1080),(1440,900)):
   page.set_viewport_size({'width':w,'height':h});page.wait_for_timeout(100);page.screenshot(path=str(out/f'{name}_{w}.png'))
 page.goto(url);page.locator('#nav-new').click();page.locator('#user').set_input_files(video);page.locator('[data-mode=none]').click();shot('upload')
 assert page.locator('#exercise').input_value()=='auto'
 with page.expect_response(lambda r:'/api/jobs' in r.url and r.request.method=='POST') as response:page.locator('#submit').click()
 jobid=response.value.json()['id'];page.wait_for_selector('#processing:not([hidden])');shot('processing')
 deadline=time.monotonic()+600
 while True:
  job=page.request.get(url+'/api/jobs/'+jobid).json()
  if job['status'] in ('needs_confirmation','failed','completed'):break
  assert time.monotonic()<deadline;page.wait_for_timeout(500)
 assert job['status']=='needs_confirmation',job
 page.wait_for_selector('#confirm-panel:not([hidden])');shot('confirm')
 page.locator('[data-confirm=squat]').click();deadline=time.monotonic()+180
 while '/sessions/' not in page.url:
  state=page.request.get(url+'/api/jobs/'+jobid).json()
  if state['status']=='failed':raise RuntimeError(state)
  assert time.monotonic()<deadline;page.wait_for_timeout(500)
 page.wait_for_selector('#exercise-chip');shot('report');assert '직접 확인' in page.locator('#exercise-chip').inner_text()
 page.goto(url);page.wait_for_selector('.session-link');shot('history')
 session=page.request.get(url+'/api/sessions').json()[0];assert session['exercise_source']=='user_confirmed_auto'
 (out/'checks.json').write_text(json.dumps({'job':jobid,'pending':job,'session':session,'console_errors':errors},ensure_ascii=False,indent=2));assert not errors,errors
 browser.close()
