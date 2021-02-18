import os
import glob
from natsort import natsorted


def remove_pickle():
    root_dir = "/home/tyler/work/data/GeekInt/data_dc/hor02"
    # root_dir = "/mnt/ai_filestore/home/tyler/dense_corr/data_dc/hor02"
    paths = natsorted(glob.glob(os.path.join(root_dir, "*", "*", "*.pkl")))
    print(len(paths))
    print(paths)

    for path in paths:
        os.remove(path)
    return


if __name__ == "__main__":
    remove_pickle()
