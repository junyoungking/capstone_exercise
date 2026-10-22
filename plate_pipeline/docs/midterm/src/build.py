"""Make one self-contained demo page: bundle data, three.js and subset fonts inlined (works offline)."""
import base64, io, json, re, sys
from fontTools.ttLib import TTCollection
from fontTools import subset

D = sys.argv[1]
B = json.load(open(f"{D}/data/bundle2.json"))
for k in ("user", "ref"):
    for f in ("skel", "h"):
        B[k].pop(f, None)
s = open(f"{D}/template.html").read()
js = json.dumps(B, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
assert s.count("/*__BUNDLE__*/null") == 1
s = s.replace("/*__BUNDLE__*/null", js)

three = open(f"{D}/vendor/package/build/three.min.js").read()
tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
assert s.count(tag) == 1 and "</script" not in three
s = s.replace(tag, "<script>" + three + "</script>")

# fonts: drop Google Fonts links, embed subsets
s = re.sub(r'<link rel="preconnect"[^>]*>\n', "", s)
s = re.sub(r'<link href="https://fonts.googleapis.com[^>]*>\n', "", s)
chars = set(s) | set(" 0123456789.,:;%/()[]+-–—·×→°?!'\"")
text = "".join(sorted(c for c in chars if ord(c) >= 32))


def sub(path, index, keep):
    font = TTCollection(path).fonts[index]
    opt = subset.Options(); opt.flavor = "woff2"; opt.layout_features = ["*"]; opt.name_IDs = ["*"]
    ss = subset.Subsetter(opt); ss.populate(text=keep); ss.subset(font)
    buf = io.BytesIO(); font.flavor = "woff2"; font.save(buf)
    return base64.b64encode(buf.getvalue()).decode()


faces = []
for w, name in [(400, "Regular"), (500, "Medium"), (700, "Bold"), (900, "Black")]:
    b64 = sub(f"/usr/share/fonts/opentype/noto/NotoSansCJK-{name}.ttc", 1, text)
    faces.append(f'@font-face{{font-family:"Noto Sans KR";font-weight:{w};font-style:normal;src:url(data:font/woff2;base64,{b64}) format("woff2")}}')
for w in (500, 700):
    b64 = base64.b64encode(open(f"{D}/vendor/package/files/jetbrains-mono-latin-{w}-normal.woff2", "rb").read()).decode()
    faces.append(f'@font-face{{font-family:"JetBrains Mono";font-weight:{w};font-style:normal;src:url(data:font/woff2;base64,{b64}) format("woff2")}}')
s = s.replace("<style>\n", "<style>\n" + "\n".join(faces) + "\n", 1)
out = f"{D}/squat_compare_demo.html"
open(out, "w").write(s)
print(out, round(len(s.encode()) / 1024), "KB", "glyph chars", len(text))
