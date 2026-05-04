# 
try:
    from nudenet import NudeDetector as _NnetDetector
    nnet_instance = _NnetDetector()
    NNET_AVAILABLE = True
except Exception as _e:
    print(f"[_flagged_labels] Content classifier unavailable: {_e}")
    nnet_instance = None
    NNET_AVAILABLE = False
