import numpy as np
from mshf.core.evaluate import cluster_bootstrap_ci, paired_tests


def test_cluster_bootstrap_returns_ordered_interval():
    y=np.array([0,1,0,1,0,1]); pred=y.copy(); groups=np.array(["a","a","b","b","c","c"])
    lo,hi=cluster_bootstrap_ci(y,pred,groups,lambda a,b: float(np.mean(a==b)),n_boot=50)
    assert lo == 1.0 and hi == 1.0


def test_paired_tests_detect_direction():
    y=np.array([0,1,0,1,0,1]); a=np.zeros(6,int); b=y.copy(); groups=np.array(["a","a","b","b","c","c"])
    result=paired_tests(y,a,b,groups)
    assert result["mean_paired_difference"] > 0
