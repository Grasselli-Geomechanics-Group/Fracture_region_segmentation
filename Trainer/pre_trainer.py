import sys
from Trainer.Segment_model import Segment_model

def pre_trainer(cfg_proj, cfg_m):
    return getattr(sys.modules[__name__], cfg_proj.solver_alg)(cfg_proj, cfg_m)