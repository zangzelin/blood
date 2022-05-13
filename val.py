from glob import glob
import os
import json
import numpy as np

# accs, aucs = {}, {}
key = "FillWithMiddle"
maxAucMark, minAucMark, maxAccMark, minAccMark = 0, 0, 0, 0
maxAuc, minAuc, maxAcc, minAcc = 0, 1, 0, 1
cnt = 0
flist = []
for f in glob(os.path.join('log', '*', f'*Blod{key}', 'test.log')):
    flist.append(f)
    with open(f, 'r') as f:
        lines = [line.replace("\'", "\"").strip() for line in f.readlines()]
    res = list(json.loads(lines[0]).values())
    if res[1] > maxAuc:
        maxAuc = res[1]
        maxAucMark = cnt
    if res[1] < minAuc:
        minAuc = res[1]
        minAucMark = cnt
    if res[2] > maxAcc:
        maxAcc = res[2]
        maxAccMark = cnt
    if res[2] < minAcc:
        minAcc = res[2]
        minAccMark = cnt
    cnt += 1

print(f"maxAcc: {maxAcc}, maxAccPosition: {flist[maxAccMark]}, maxAuc: {maxAuc}, maxAucPosition: {flist[maxAucMark]}")
print(f"minAcc: {minAcc}, minAccPosition: {flist[minAccMark]}, minAuc: {minAuc}, minAucPosition: {flist[minAucMark]}")
# print(f"key: {key}, acc: {np.mean(accs)}, acc_std:{np.std(accs)}, auc: {np.mean(aucs)}")
