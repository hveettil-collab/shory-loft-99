"""Pass 1: detect display quads (inner hole of the black bezel) on every frame."""
import sys, json, cv2, numpy as np
frames_dir, out_json = sys.argv[1], sys.argv[2]

def quad_from_contour(c):
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.08, 30):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(float)
    return None

def order(q):
    # TL, TR, BR, BL
    s = q.sum(1); d = np.diff(q, axis=1).ravel()
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]])

res = {}
for i in range(1, 361):
    im = cv2.imread(f"{frames_dir}/f{i:03d}.png")
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    dark = (g < 60).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, hier = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    if hier is not None:
        for k, c in enumerate(cnts):
            if hier[0][k][3] >= 0:  # hole
                a = cv2.contourArea(c)
                if a > 12000:
                    q = quad_from_contour(c)
                    if q is not None:
                        qa = cv2.contourArea(q.astype(np.float32))
                        cands.append((a, a / max(qa, 1), order(q).tolist()))
    cands.sort(key=lambda t: -t[0])
    res[i] = [{"area": a, "fill": f, "quad": q} for a, f, q in cands[:3]]
json.dump(res, open(out_json, "w"))
ok = [i for i in res if res[i] and res[i][0]["fill"] > 0.9]
print("frames with confident quad:", len(ok))
miss = [i for i in range(1, 361) if i not in ok]
# print runs of misses
runs=[];s=None
for i in range(1,362):
    m = i in miss
    if m and s is None: s=i
    if not m and s is not None: runs.append((s,i-1)); s=None
print("miss runs:", runs)
