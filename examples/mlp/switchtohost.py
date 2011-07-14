import cuv_python as cp

def switchtohost():
    cp.dev_tensor_float_cm_orig = cp.dev_tensor_float_cm
    cp.dev_tensor_float_cm      = cp.host_tensor_float_cm
    cp.dev_tensor_float_orig = cp.dev_tensor_float
    cp.dev_tensor_float      = cp.host_tensor_float
