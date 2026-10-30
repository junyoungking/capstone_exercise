"""Offline Chromium checks: python -m eval.m3a_verify --runs DIR --out DIR."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright
from server.schema import validate_result

def verify(runs,out):
    out.mkdir(parents=True,exist_ok=True);results=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader'])
        for name in ('squat','benchpress','deadlift'):
            source=runs/name;b=validate_result(json.loads((source/'result_v3.json').read_text()))
            if b['inputs']['height_cm'] is not None:
                assert abs(b['user']['pose3d_height_cm']-b['inputs']['height_cm'])<1e-6
            assert abs(b['user']['pose3d_height_cm']-b['ref']['pose3d_height_cm'])<1e-6
            for role in ('user','ref'):
                assert b[role]['geo']['stature_cm']==b['inputs']['height_cm']
                if b['inputs']['height_cm'] is None:
                    assert all(b[role]['geo'][key] is None for key in ('thigh_cm','shin_cm','torso_cm','upper_cm','fore_cm'))
            page=browser.new_page(viewport=dict(width=1920,height=1080),offline=True)
            errors=[];requests=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
            page.on('request',lambda r:requests.append(r.url) if r.url.startswith(('http:','https:')) else None)
            page.goto((source/'report.html').resolve().as_uri())
            page.wait_for_function('document.querySelector("#v-user").readyState>=2 && document.querySelector("#v-ref").readyState>=2')
            page.evaluate('seekPos("FAR")');page.wait_for_timeout(500)
            assert page.locator('#s-report').is_visible() and not page.locator('#s-upload').count()
            assert page.locator('#pose2d').count()==(0 if b['show_3d'] else 1)
            assert page.locator('#three canvas').count()==1
            shots=[]
            for label,selector in (('top',None),('cards','#fb-list'),('pose','#three')):
                if selector: page.locator(selector).scroll_into_view_if_needed()
                else: page.evaluate('scrollTo(0,0)')
                page.wait_for_timeout(150);path=out/f'{name}_{label}.png';page.screenshot(path=str(path));shots.append(path.name)
            expression='({geo:document.querySelector("#geo").innerText,rom:rom(U.reps[0]),v:vel(U.reps[0]),depth:depthAt(U,U.reps[0].far),q:JSON.stringify(U.j3),k:JSON.stringify(B.kpi),cam:T3 ? T3.mine.g.scale.x : 1})'
            before=page.evaluate(expression)
            page.locator('#r-plate').fill(str(b['inputs']['plate_mm']*.8));page.locator('#r-plate').dispatch_event('change')
            after=page.evaluate(expression)
            for k in ('geo','q','k','cam'): assert before[k]==after[k],(name,k)
            for k in ('rom','v','depth'): assert abs(before[k]*.8-after[k])<1e-8,(name,k)
            if b['findings']['cards']: page.locator('.fcard .go').first.click()
            page.locator('#posbtns select').select_option(str(len(b['ref']['reps'])))
            page.locator('#pp').click();page.wait_for_timeout(300);page.locator('#pp').click()
            assert not errors,(name,errors)
            assert not requests,(name,requests)
            results.append(dict(exercise=name,viewport=[1920,1080],errors=errors,network_requests=requests,diameter_ratio=.8,plate_values_scale=True,geo_and_3d_unchanged=True,captures=shots));page.close()
        browser.close()
    (out/'browser_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));return results

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();print(json.dumps(verify(a.runs,a.out),ensure_ascii=False,indent=2))
