import sys, json, cv2, numpy as np
frames_dir, js, out, step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
a, b = int(sys.argv[5]), int(sys.argv[6])
d = json.load(open(js))
tiles = []
for i in range(a, b, step):
    im = cv2.imread(f"{frames_dir}/f{i:03d}.png")
    for k, c in enumerate(d.get(str(i), [])[:2]):
        q = np.array(c["quad"], np.int32)
        cv2.polylines(im, [q], True, (0, 0, 255) if k == 0 else (255, 0, 0), 2)
    cv2.putText(im, str(i), (10, 40), 0, 1.2, (0, 0, 255), 3)
    tiles.append(cv2.resize(im, (240, 427)))
while len(tiles) % 10: tiles.append(np.full_like(tiles[0], 255))
rows = [np.hstack(tiles[r:r + 10]) for r in range(0, len(tiles), 10)]
cv2.imwrite(out, np.vstack(rows))
