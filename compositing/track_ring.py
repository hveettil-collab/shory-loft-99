"""Bezel-ring tracker for stretches where the screen CONTENT changes (iOS blur, scrolling, video).
Tracks KLT points on the black bezel ring + punch-hole only (skin excluded) and fits a
4-DOF similarity per frame. Usage: track_ring.py frames in.json out.json start end anchor"""
import sys, json, cv2, numpy as np
fd, inj, outj, a, b, anchor = sys.argv[1], sys.argv[2], sys.argv[3], *map(int, sys.argv[4:7])
data = json.load(open(inj))
def img(i): return cv2.imread(f"{fd}/f{i:03d}.png")
def skin(im):
    ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    sk = cv2.inRange(ycc, (40, 138, 80), (240, 175, 125))
    return cv2.dilate(cv2.morphologyEx(sk, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((21, 21), np.uint8))
def ring(q, shape):
    q = np.float32(q); c = q.mean(0); m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, ((q - c) * 1.09 + c).astype(np.int32), 255)
    cv2.fillConvexPoly(m, ((q - c) * 0.95 + c).astype(np.int32), 0)
    return m
quad = np.float32(data[str(anchor)]["quad"] if "quad" in data[str(anchor)] else data[str(anchor)][0]["quad"])
out = {}
for step in (1, -1):
    q = quad.copy(); prev = img(anchor); pg = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    rng = range(anchor + 1, b + 1) if step == 1 else range(anchor - 1, a - 1, -1)
    for i in rng:
        cur = img(i); cg = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        m = ring(q, pg.shape); m[skin(prev) > 0] = 0
        p0 = cv2.goodFeaturesToTrack(pg, 300, 0.01, 4, mask=m)
        if p0 is None or len(p0) < 6: print("lost at", i); break
        p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, cg, p0, None, winSize=(21, 21), maxLevel=3)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(cg, pg, p1, None, winSize=(21, 21), maxLevel=3)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pb - p0, axis=2).ravel() < 1.0)
        if ok.sum() < 6: print("lost at", i); break
        M, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        q = cv2.transform(q[None], M)[0]
        out[i] = {"quad": q.tolist(), "src": "ring", "inliers": int(inl.sum())}
        prev, pg = cur, cg
out[anchor] = {"quad": quad.tolist(), "src": "anchor"}
json.dump(out, open(outj, "w")); print("ring-tracked", len(out), "frames", min(map(int, out)), "-", max(map(int, out)))
