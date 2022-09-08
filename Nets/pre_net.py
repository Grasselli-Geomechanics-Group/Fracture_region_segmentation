import sys
from Nets.network import U_Net, R2U_Net


def pre_net(cfg_proj, cfg_m):
    return getattr(sys.modules[__name__], cfg_proj.backbone)(img_ch = cfg_m.img_ch, output_ch = cfg_m.output_ch)