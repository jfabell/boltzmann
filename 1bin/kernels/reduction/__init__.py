reduction = None

def init_reduction(pyk: bool):
    global reduction
    if pyk:
        from . import pyk_reduction as reduction_impl
        reduction = reduction_impl.pyk_reduction_fast
        #reduction = reduction_impl.reduction
    else:
        from . import cuda_reduction as reduction_impl
        reduction = reduction_impl.reduction
