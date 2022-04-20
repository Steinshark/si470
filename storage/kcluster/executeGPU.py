from DataTools import *
import sys
from matplotlib import pyplot as plt
_GPU_MODE = False

if __name__ == "__main__":
    if sys.argv[1] == 'GPU':
        import cupy
        _GPU_MODE = True

    matr, docs = load_data(read_npz=os.path.isfile('preSVD.npz'),saving_npz=True)

    n_comp = [1,2,3,4,5,6,7,8,9,10]

    n_var = []
    n_time = []
    for n in n_comp:
        newMatr, model,time = svd_decomp(n=n,matrix=matr)
        n_var.append(model.explained_variance_ratio_.sum())
        print("\tN complete")
        n_time.append(time/60)
    plt.scatter(n_comp,n_var)
    plt.scatter(n_comp,n_time)
    plt.show()
