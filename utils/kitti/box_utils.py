import numpy as np
import torch


class BoxCoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = kwargs.get('encode_angle_by_sincos', False)
        if self.encode_angle_by_sincos:
            self.code_size += 1


    def encode_torch(self, boxes, anchors):
        """
        Args:
            y -> height
            boxes: (N, 7 + C) [x, y, z, h, w, l, heading, ...]
            anchors: (N, 7 + C) [dx, dy, dz, dh, dw, dl, dheading, ...]

        Returns:

        """
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dha, dwa, dla, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dhg, dwg, dlg, rg, *cgs = torch.split(boxes[:, :-1], 1, dim=-1)

        diagonal = torch.sqrt(dwa ** 2 + dla ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / dha
        zt = (zg - za) / diagonal
        dht = torch.log(dhg / dha)
        dwt = torch.log(dwg / dwa)
        dlt = torch.log(dlg / dla)
       
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
            rt = torch.cat(rts, dim=-1)
        else:
            rt = rg - ra
        
            rt[rt > np.pi - np.pi / 4] -= np.pi
            reverse_flag = rt > np.pi / 4
            rt[reverse_flag] -= np.pi / 2
            if reverse_flag.any():
                dwt[reverse_flag] = torch.log(dlg[reverse_flag] / dwa[reverse_flag])
                dlt[reverse_flag] = torch.log(dwg[reverse_flag] / dwa[reverse_flag])        
        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dht, dwt, dlt, rt, *cts], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dha, dwa, dla, ra, *cas = torch.split(anchors, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dht, dwt, dlt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dht, dwt, dlt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dwa ** 2 + dla ** 2)
        xg = xt * diagonal + xa
        yg = yt * dha + ya
        zg = zt * diagonal + za

        dhg = torch.exp(dht) * dha
        dwg = torch.exp(dwt) * dwa
        dlg = torch.exp(dlt) * dla

        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]

        return torch.cat([xg, yg, zg, dhg, dwg, dlg, rg, *cgs], dim=-1)
