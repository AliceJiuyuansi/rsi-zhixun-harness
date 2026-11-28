"""A-only split of the authorized training projection. Equal inputs stay together."""
import hashlib
import json
import numpy as np


def partition(data):
    held = np.array([int.from_bytes(hashlib.sha256(np.asarray(x + 0., dtype='<f8').tobytes()).digest(), 'big') % 5 == 0 for x in data['x']])
    if not held.any() or held.all():
        return data, data, False
    def subset(mask):
        idx = np.flatnonzero(mask)
        ids = [data['ids'][i] for i in idx]
        proof = json.dumps([data['sha256'], ids], ensure_ascii=False, separators=(',', ':')).encode()
        return dict(x=data['x'][idx], y=data['y'][idx], ids=ids, groups=[data['groups'][i] for i in idx], paths=data['paths'], sha256='sha256:' + hashlib.sha256(proof).hexdigest())
    return subset(~held), subset(held), True
