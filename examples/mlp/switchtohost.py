import cuv_python as cp

def switchtohost():
    cp.dev_matrix_cmf_orig = cp.dev_matrix_cmf
    cp.dev_matrix_cmf      = cp.host_matrix_cmf
    cp.push                = cp.push_host
    cp.pull                = cp.pull_host
