import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import os
import json
from collections import OrderedDict

PI = float(np.pi)

def resize_crop(img, scale, size):
    
    re_size = int(img.shape[0]*scale)

    if(re_size>0):
        img = cv2.resize(img, (re_size, re_size), cv2.INTER_AREA)

    if size <= re_size:
        pd = int((re_size-size)/2)
        img = img[pd:pd+size,pd:pd+size]
    else:
        new = np.zeros((size,size))
        pd = int((size-re_size)/2)
        new[pd:pd+re_size,pd:pd+re_size] = img[:,:]
        img = new

    return img

def resize(img, scale):
    
    re_size = int(img.shape[0]*scale)
    if(re_size>0):
        img = cv2.resize(img, (re_size, re_size), cv2.INTER_CUBIC)
        
    return img

def var2np(data_lst):

    def trans(data):
        if data.shape[1] == 1:
            return data[0, 0].data.cpu().numpy()
        elif data.shape[1] == 3: 
            return data[0, :, :, :].permute(1, 2, 0).data.cpu().numpy()

    if isinstance(data_lst, list):
        np_lst = []
        for data in data_lst:
            np_lst.append(trans(data))
        return np_lst
    else:
        return trans(data_lst)


###find the the largest connected component - evaluate ceiling reliability for special cases (eg curved ceilings): using floor shape in case of ureliable ceiling
    ###conditions: 1. best connected component must be times larger than others (using threshold) 2. must contain the camera
def approx_shape(data, fp_threshold=0.5, epsilon_b=0.005, rel_threshold=0.5, return_reliability=False):
    data_c = data.copy()
    ret, data_thresh = cv2.threshold(data_c, fp_threshold, 1, 0)
    data_thresh = np.uint8(data_thresh)

    w = data.shape[0]
    h = data.shape[1] 
    
    data_cnt, data_heri = cv2.findContours(data_thresh, 1, 2)##CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE
    ##data_cnt, data_heri = cv2.findContours(data_thresh, 0, 2)##CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE
        
    reliability = 0.0

    approx = np.empty([1, 1, 2])
    
    if(len(data_cnt)>0):    
        # Find the the largest connected component and its bounding box
        data_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)
        
        area0 = cv2.contourArea(data_cnt[0])

        if(len(data_cnt)>1 and (area0 > 0)):
            area1 = cv2.contourArea(data_cnt[1])
            ###condition 1.
            reliability = 1.0 - (area1/area0)
        else:
            reliability = 1.0

        ##print('connected components', len(data_cnt))
                 
    if(reliability<rel_threshold and len(data_cnt)>1):
        mergedlist = np.concatenate((data_cnt[0], data_cnt[1]), axis=0)
        approx = cv2.convexHull(mergedlist)

        ###condition 2.: check camera
        dist = cv2.pointPolygonTest(approx,(w/2,h/2), True)

        if(dist<0):
            reliability = 0.1

    else:
        
        if(len(data_cnt)>0):
            epsilon = epsilon_b*cv2.arcLength(data_cnt[0], True)
            approx = cv2.approxPolyDP(data_cnt[0], epsilon, True)
            ##reliability = 0.0

            ###condition 2.: check camera
            dist = cv2.pointPolygonTest(approx,(w/2,h/2), True)

            if(dist<0):
                reliability = 0.1


    if return_reliability:
        ap_area = cv2.contourArea(approx)
        return approx,reliability,ap_area
    else:
        return approx

def metric_scale(height, camera_h, fp_meter, fp_size):
        
    scale = 100 * ((height - camera_h) / camera_h) * (fp_meter / fp_size)
        
    return scale

def x2image(x):
    img = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)

    return img

def recover_h_value(mask):
    return np.amax(mask.numpy())

def load_trained_model(Net, path):
    state_dict = torch.load(path, map_location='cpu')
    net = Net(**state_dict['kwargs'])
    net.load_state_dict(state_dict['state_dict'])
    return net

def export2json(c_pts, W, H, fp_size, output_dir,def_img,k, z0, z1):
    c_pts = c_pts.squeeze(1)       
                             
    c_cor = np_xy2coor(np.array(c_pts), z0, W, H, fp_size, fp_size)
    f_cor = np_xy2coor(np.array(c_pts), z1, W, H, fp_size, fp_size) ####based on the ceiling shape

    cor_count = len(c_cor)                                                        
            
    c_ind = np.lexsort((c_cor[:,1],c_cor[:,0])) 
    f_ind = np.lexsort((f_cor[:,1],f_cor[:,0]))
            
    ####sorted by theta (pixels coords)
    c_cor = c_cor[c_ind]
    f_cor = f_cor[f_ind]
                       
    cor_id = []

    for j in range(len(c_cor)):
        cor_id.append(c_cor[j])
        cor_id.append(f_cor[j])
   
    cor_id = np.array(cor_id)
              
                                    
    # Normalized to [0, 1]
    cor_id[:, 0] /= W
    cor_id[:, 1] /= H
                        
    # Output result
    with open(os.path.join(output_dir, k + '.json'), 'w') as f:
        json.dump({
        'z0': float(z0),
        'z1': float(z1),
        'uv': [[float(u), float(v)] for u, v in cor_id],
        }, f)
            
    ##store json full path name for 3D visualization - NB only uv and z are stored
    json_name = os.path.join(output_dir, k + '.json')

    return json_name

def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y

def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi

def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi

def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5

def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    if p1[0] == p2[0]:
        return np.array([p1, p2], np.float32)

    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)
    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)
    coorys = v2coory(vs)

    return np.stack([coorxs, coorys], axis=-1)    

def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys

def visualize_equi_model(x, y_bon):
    x = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
               
    y_bon = y_bon.numpy()
    y_bon = ((y_bon / np.pi + 0.5) * x.shape[0]).round().astype(int)
        
    img_bon = (x.copy()).astype(np.uint8)

    img_bon_empty = np.zeros([512,1024,3],dtype=np.uint8)
        
    h = x.shape[0]
    w = x.shape[1]
           
    img_bon[y_bon[0], np.arange(len(y_bon[0])), 2] = 255
    img_bon[y_bon[1], np.arange(len(y_bon[1])), 0] = 255

    img_bon_empty[y_bon[0], np.arange(len(y_bon[0])), 2] = 255
    img_bon_empty[y_bon[1], np.arange(len(y_bon[1])), 0] = 255
    
    h, w = img_bon.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    mask[:] = 0

    h_pt = int(w/2), int(h/2)
    c_pt = int(w/2), 1
    f_pt = int(w/2), h-1
        
    kernel = np.ones((5,5),np.uint8)
    img_bon_empty = cv2.dilate(img_bon_empty,kernel,iterations = 2)

    cv2.floodFill(img_bon_empty, mask, f_pt, (96,0,0))
    cv2.floodFill(img_bon_empty, mask, c_pt, (0,0,96))
    cv2.floodFill(img_bon_empty, mask, h_pt, (0,96,0))

    img_bon = cv2.add(img_bon,img_bon_empty)
    ##img_bon += img_bon_empty

    return img_bon

def transform2equi(c_pts,h_c_mean,h_f_mean, W, H, fp_size,scale):
    c_pts = c_pts.squeeze(1)   
                                       
    c_cor = np_xy2coor(np.array(c_pts), h_c_mean, W, H, fp_size*scale, fp_size*scale)
    f_cor = np_xy2coor(np.array(c_pts), -h_f_mean, W, H, fp_size*scale, fp_size*scale) ####based on the ceiling shape

    cor_count = len(c_cor)                                                        
            
    c_ind = np.lexsort((c_cor[:,1],c_cor[:,0])) 
    f_ind = np.lexsort((f_cor[:,1],f_cor[:,0]))
            
    ####sorted by theta (pixels coords)
    c_cor = c_cor[c_ind]
    f_cor = f_cor[f_ind]
                       
    cor_id = []

    for j in range(len(c_cor)):
        cor_id.append(c_cor[j])
        cor_id.append(f_cor[j])
   
    cor_id = np.array(cor_id)         
                                             
    cor = np.roll(cor_id[:, :2], -2 * np.argmin(cor_id[::2, 0]), 0)

    ##print('cor shape',cor.shape)

    # Prepare 1d ceiling-wall/floor-wall boundary
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []

    n_cor = len(cor)
                                        
    for i in range(n_cor // 2): 
        xys = pano_connect_points(cor[i*2],cor[(i*2+2) % n_cor],z=-50)
        bon_ceil_x.extend(xys[:, 0])          
        bon_ceil_y.extend(xys[:, 1])

                ##print('ceiling list',len(bon_ceil_x),len(bon_ceil_y))

    draw_floor_mask = True

    if draw_floor_mask:
                for i in range(n_cor // 2): 
                        ##NB expecting corner coords in pixel
                        xys = pano_connect_points(cor[i*2+1],
                                                    cor[(i*2+3) % n_cor],
                                                    z=50)            
                        bon_floor_x.extend(xys[:, 0])            
                        bon_floor_y.extend(xys[:, 1])
                else:
                ##NB using only ceiling shape
                    for i in range(n_cor // 2): 
                        ###NB expecting corner coords in pixel
                        xys = pano_connect_points(cor[i*2],
                                                    cor[(i*2+2) % n_cor],
                                                    z=50)            
                        bon_floor_x.extend(xys[:, 0])            
                        bon_floor_y.extend(xys[:, 1])
                
        
    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)        
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)        
                
    bon = np.zeros((2, W))        
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)        
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)
                                                                       
    ###normalize to image height (from px to 0-1)
    bon = ((bon + 0.5) / H - 0.5) * np.pi

    return bon  

def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * PI


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * PI

def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512, m_ratio = 1.0):
    '''
    coor: N x 2, index of array in (col, row) format eg. 1024x2
    m_ratio: pixel/cm ratio for tensor fitting
    '''           
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)       
    v = np_coory2v(coor[:, 1], coorH)
                  
    c = z / np.tan(v)
    x = m_ratio * c * np.sin(u) + floorW / 2 - 0.5
    y = -m_ratio *c * np.cos(u) + floorH / 2 - 0.5
    
    return np.hstack([x[:, None], y[:, None]])

def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi


def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi


def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5


def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y

def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    if p1[0] == p2[0]:
        return np.array([p1, p2], np.float32)

    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)
    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)
    coorys = v2coory(vs)

    return np.stack([coorxs, coorys], axis=-1)

def np_xy2coor(xy, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    x = xy[:, 0] - floorW / 2 + 0.5
    y = xy[:, 1] - floorH / 2 + 0.5

    u = np.arctan2(x, -y)
    v = np.arctan(z / np.sqrt(x**2 + y**2))

    coorx = (u / (2 * PI) + 0.5) * coorW - 0.5
    coory = (-v / PI + 0.5) * coorH - 0.5

    return np.hstack([coorx[:, None], coory[:, None]])


               
          
     





                
     