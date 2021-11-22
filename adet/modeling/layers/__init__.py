
try:
    from .quantization.layers import actv, norm, shuffle, split, concat, add
    from .quantization.quant  import qconv, qlinear
except:
    # fall back to detectron2 interface
    pass

