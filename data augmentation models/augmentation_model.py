import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

    
def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=x.shape)
    return np.multiply(x, factor)

    
def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(len(x))
    
    num_segs = np.random.randint(1, max_segments, size=(len(x)))
    
    ret = np.zeros_like(x)
    for i, val in enumerate(x):
        if isinstance(val, float):
            ret[i] = val
        elif num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(len(x) - 1, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = val[warp]
        else:
            ret[i] = val
    return ret


