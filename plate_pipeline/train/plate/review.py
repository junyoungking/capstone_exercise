"""Box review tool (OpenCV window): fix the automatic boxes by hand.

python -m train.plate.review --labels data/own/auto.json --images data/own/images --out data/own/labels.json

Keys
  drag (left mouse)   draw a new box
  click inside a box  select it (red)          d / Delete   delete the selected box
  a / Enter / Space   accept this image, next  n / p        next / previous without marking
  u                   mark "unusable" (excluded from the dataset)
  s                   save                     q / Esc      save and quit
Images are shown lowest-confidence first among the ones not reviewed yet. Progress is saved in the
output file ("reviewed": true on each image), so you can stop and continue.
Label rule: every plate at least half visible; the box goes around the whole rim, not the bar end.
"""
import argparse
import os

import cv2

from train.plate import coco

WIN = "plate review"


class Review:
    def __init__(self, d, img_dir, out):
        self.d, self.img_dir, self.out = d, img_dir, out
        by = {}
        for a in d["annotations"]:
            by.setdefault(a["image_id"], []).append(a)
        self.by = by
        todo = [im for im in d["images"] if not im.get("reviewed")]
        done = [im for im in d["images"] if im.get("reviewed")]
        todo.sort(key=lambda im: min([a.get("score", 1.0) for a in by.get(im["id"], [])] or [0.0]))
        self.order = todo + done
        self.i, self.sel, self.drag = 0, None, None

    def boxes(self, im):
        return self.by.setdefault(im["id"], [])

    def scale(self, im):
        return min(1.0, 1400 / max(im["width"], im["height"]))

    def draw(self):
        im = self.order[self.i]
        img = cv2.imread(os.path.join(self.img_dir, im["file_name"]))
        k = self.scale(im)
        img = cv2.resize(img, None, fx=k, fy=k)
        for j, a in enumerate(self.boxes(im)):
            x, y, w, h = (v * k for v in a["bbox"])
            col = (0, 0, 255) if j == self.sel else (0, 220, 0)
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), col, 2)
            if "score" in a:
                cv2.putText(img, f"{a['score']:.2f}", (int(x), int(y) - 4), 0, 0.6, col, 2)
        if self.drag and self.drag[2] is not None:
            (x0, y0), (x1, y1) = self.drag[:2], self.drag[2]
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 180, 0), 2)
        n_done = sum(1 for m in self.d["images"] if m.get("reviewed"))
        tag = "UNUSABLE" if im.get("unusable") else ("reviewed" if im.get("reviewed") else "")
        cv2.putText(img, f"{self.i + 1}/{len(self.order)}  done {n_done}  {im['file_name']}  {tag}", (10, 24), 0,
                    0.6, (0, 255, 255), 2)
        cv2.imshow(WIN, img)

    def mouse(self, ev, x, y, flags, _):
        im = self.order[self.i]
        k = self.scale(im)
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drag = [x, y, None]
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            self.drag[2] = (x, y)
            self.draw()
        elif ev == cv2.EVENT_LBUTTONUP and self.drag:
            x0, y0 = self.drag[:2]
            self.drag = None
            if abs(x - x0) > 8 and abs(y - y0) > 8:
                box = (min(x0, x) / k, min(y0, y) / k, max(x0, x) / k, max(y0, y) / k)
                aid = coco.add_box(self.d, im["id"], box)
                self.boxes(im).append(self.d["annotations"][-1])
                self.sel = len(self.boxes(im)) - 1
            else:
                self.sel = None
                for j, a in enumerate(self.boxes(im)):
                    bx, by, bw, bh = a["bbox"]
                    if bx <= x / k <= bx + bw and by <= y / k <= by + bh:
                        self.sel = j
            self.draw()

    def delete(self):
        im = self.order[self.i]
        bs = self.boxes(im)
        if self.sel is not None and self.sel < len(bs):
            a = bs.pop(self.sel)
            self.d["annotations"] = [b for b in self.d["annotations"] if b["id"] != a["id"]]
            self.sel = None

    def save(self):
        coco.save(self.d, self.out)

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.mouse)
        while True:
            self.draw()
            key = cv2.waitKeyEx(0)
            im = self.order[self.i]
            if key in (ord("q"), 27):
                break
            if key in (ord("a"), 13, 32):
                im["reviewed"] = True
                im.pop("unusable", None)
                self.i = min(self.i + 1, len(self.order) - 1)
                self.sel = None
            elif key == ord("u"):
                im["reviewed"] = True
                im["unusable"] = True
                self.i = min(self.i + 1, len(self.order) - 1)
            elif key == ord("n"):
                self.i = min(self.i + 1, len(self.order) - 1)
                self.sel = None
            elif key == ord("p"):
                self.i = max(self.i - 1, 0)
                self.sel = None
            elif key in (ord("d"), 0x2E0000, 127, 8):
                self.delete()
            elif key == ord("s"):
                self.save()
        self.save()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = a.out if os.path.exists(a.out) else a.labels
    Review(coco.load(src), a.images, a.out).run()


if __name__ == "__main__":
    main()
