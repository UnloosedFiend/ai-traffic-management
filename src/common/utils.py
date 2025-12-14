# src/common/utils.py
import numpy as np

def xywh2xyxy(x, y, w, h):
    # convert center x,y,w,h normalized to x1,y1,x2,y2 (normalized)
    x1 = x - w/2.0
    y1 = y - h/2.0
    x2 = x + w/2.0
    y2 = y + h/2.0
    return x1, y1, x2, y2

def scale_boxes_norm_to_pixel(box, img_w, img_h):
    # box = [x1,y1,x2,y2] normalized (0..1), returns pixel coords as ints
    x1 = int(max(0, min(img_w-1, round(box[0]*img_w))))
    y1 = int(max(0, min(img_h-1, round(box[1]*img_h))))
    x2 = int(max(0, min(img_w-1, round(box[2]*img_w))))
    y2 = int(max(0, min(img_h-1, round(box[3]*img_h))))
    return x1, y1, x2, y2
