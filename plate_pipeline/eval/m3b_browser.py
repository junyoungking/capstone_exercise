"""Real local upload flow; no fake analysis. Captures and stage observations.

python -m eval.m3b_browser --url http://127.0.0.1:8765 --out results/m3b
Requires the team's local videos and an empty application database.
"""
import argparse,json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
LOCAL=Path(__file__).resolve().parents[1]/'data/team/test9'


def run(url,out,videos=LOCAL):
    out.mkdir(parents=True,exist_ok=True);checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport=dict(width=1920,height=1080));errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
        def shot(name):page.screenshot(path=str(out/(name+'.png')))
        page.goto(url);page.wait_for_selector('#empty:not([hidden])');shot('01_empty')
        def upload(name,user,exercise,ref=None,stored=False):
            print('upload',name,flush=True)
            if not page.locator('#nav-new').count():page.goto(url+'/#new')
            else:page.locator('#nav-new').click()
            page.wait_for_selector('#new:not([hidden])')
            page.locator('#user').set_input_files(str(user));page.locator('#exercise').select_option(exercise)
            page.locator('#height-cm').fill('175')
            page.locator('#ref-mode').select_option('saved' if stored else 'upload' if ref else 'none')
            if ref:page.locator('#ref').set_input_files(str(ref))
            if stored:page.locator('#ref-id').select_option(index=0)
            shot(name+'_input')
            with page.expect_response(lambda r:'/api/jobs' in r.url and r.request.method=='POST') as response:
                page.locator('#submit').click()
            resp=response.value
            assert resp.status==202,resp.text()
            id=resp.json()['id'];samples=[];captured=False;started=time.monotonic()
            while True:
                job=page.request.get(url+'/api/jobs/'+id).json();samples.append(job)
                if not captured and job['stage'] in ('plate_user','plate_ref','pose') and page.locator('#processing').is_visible():
                    shot(name+'_processing');captured=True
                if job['status'] in ('completed','failed'):break
                assert time.monotonic()-started<240
                page.wait_for_timeout(300)
            assert job['status']=='completed',job
            for k in job['stages']:
                values=[s['stages'][k]['progress'] for s in samples];assert values==sorted(values),(name,k)
            page.wait_for_url('**/sessions/'+id+'/',timeout=15000)
            page.wait_for_function('document.querySelector("#v-user").readyState>=2')
            page.evaluate('seekPos("FAR")');page.wait_for_timeout(600);shot(name+'_report')
            page.locator('#fb-list').scroll_into_view_if_needed();shot(name+'_cards')
            assert not page.locator('#ptable .k1, #ptable .k2').count()
            assert '속도 저하' in page.locator('#kpis').inner_text()
            assert page.locator('#vb-ref').is_visible()==bool(ref or stored)
            if not ref and not stored:assert page.locator('#r-title').inner_text().endswith('기록 리포트')
            result=page.request.get(url+'/sessions/'+id+'/result_v3.json').json()
            assert not any(c['key'] in ('knee','hip','trunk') for c in result['findings']['cards']+result['findings']['extra'])
            checks.append(dict(name=name,id=id,job=job,samples=samples,console_errors=list(errors)))
            print('done',name,'cache',job.get('ref_cache_hit'),flush=True)
            return id
        first=upload('02_squat',videos/'squat/stu8_67.mp4','squat',videos/'squat/stu10_70.mp4')
        page.goto(url);page.wait_for_selector('.session-link');shot('03_history')
        page.locator('.day-btn.has').click();page.wait_for_url('**/sessions/'+first+'/');shot('04_reopened')
        upload('05_deadlift',videos/'deadlift/deadlift_08.mp4','deadlift')
        upload('06_cached',videos/'squat/stu8_67.mp4','squat',stored=True)
        assert checks[-1]['job']['ref_cache_hit'] is True
        page.goto(url);page.wait_for_function('document.querySelectorAll(".session-link").length===3');shot('07_history_three')
        assert not errors,errors
        (out/'browser_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
        browser.close()
    return checks

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--url',default='http://127.0.0.1:8000');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--videos',type=Path,default=LOCAL);a=ap.parse_args();run(a.url,a.out,a.videos)
