"""Laptop-mode upper panel: screen = large saturated-blue region (+ enclosed figure), quad from hull."""
import sys, json, cv2, numpy as np
frames_dir, out_json, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def order(q):
    s = q.sum(1); d = np.diff(q, axis=1).ravel()
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]])
res = {}
for i in range(a, b + 1):
    im = cv2.imread(f"{frames_dir}/f{i:03d}.png")
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] > 95) & (hsv[..., 0] < 130) & (hsv[..., 1] > 70) & (hsv[..., 2] > 50)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: continue
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    if st[k, cv2.CC_STAT_AREA] < 8000: continue
    c, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(c, key=cv2.contourArea)); peri = cv2.arcLength(hull, True)
    q = None
    for eps in np.linspace(0.01, 0.1, 40):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4: q = ap.reshape(4, 2).astype(float); break
    if q is None: continue
    res[i] = [{"area": float(st[k, cv2.CC_STAT_AREA]), "fill": 1.0, "quad": order(q).tolist()}]
json.dump(res, open(out_json, "w")); print("laptop quads:", len(res), "of", b - a + 1)
