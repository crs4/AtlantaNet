import os
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon

from tools import np_coor2xy, np_coory2v


def test_shape(id, dt_cor_id, dt_z0, dt_z1, gt_cor_id, w, h, losses):
     
    dt_floor_coor = dt_cor_id[1::2]
    dt_ceil_coor = dt_cor_id[0::2]
    gt_floor_coor = gt_cor_id[1::2]
    gt_ceil_coor = gt_cor_id[0::2]
    assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0
      
    
    # Eval 3d IoU and height error(in meter)
    N = len(dt_floor_coor)
    
    dt_floor_xy = np_coor2xy(dt_floor_coor, dt_z1, 1024, 512, floorW=1024, floorH=1024)
    gt_floor_xy = np_coor2xy(gt_floor_coor, dt_z1, 1024, 512, floorW=1024, floorH=1024)

    ###new evaluating on ceiling
    dt_ceiling_xy = np_coor2xy(dt_ceil_coor, dt_z0, 1024, 512, floorW=1024, floorH=1024)
    gt_ceiling_xy = np_coor2xy(gt_ceil_coor, dt_z0, 1024, 512, floorW=1024, floorH=1024)

    ##print(gt_floor_xy)
            
    ##dt_poly = Polygon(dt_floor_xy)    ####NB polygon convert xy coords to image coords (1024x1024)
    ##gt_poly = Polygon(gt_floor_xy)    ####NB polygon convert xy coords to image coords (1024x1024)

    ###NB now using ceiling shape for evaluation
    dt_poly = Polygon(dt_ceiling_xy)    ####NB polygon convert xy coords to image coords (1024x1024)
    gt_poly = Polygon(gt_ceiling_xy)    ####NB polygon convert xy coords to image coords (1024x1024)

    is_valid = True
           
    if not gt_poly.is_valid:
        print('Skip ground truth invalid (%s)' % gt_path )
        is_valid = False
        ##return
        
    if not dt_poly.is_valid:
        print('Skip inferred invalid (%s)' % dt_path )
        is_valid = False
        ##return

    iou2d = 0
    iouH = 0

    if(is_valid):
        area_dt = dt_poly.area
        area_gt = gt_poly.area

        h_ratio = area_dt/area_gt
                
        area_inter = 0.0
    
        if(h_ratio>0.0):
            area_inter = dt_poly.intersection(gt_poly).area
        else:
            print('invalid area for ',gt_path,'with area dt gt',area_dt,area_gt)
            is_valid = False    
    
        iou2d = area_inter / (area_gt + area_dt - area_inter)

        cch_dt = dt_z0
            
        cch_gt = get_z1(gt_floor_coor[:, 1], gt_ceil_coor[:, 1], dt_z1, 512) ###NB actually returns z0! 
        
        h_dt = abs(cch_dt.mean() - dt_z1) ###layout height
        h_gt = abs(cch_gt.mean() - dt_z1) ###layout height

        
        iouH = min(h_dt, h_gt) / max(h_dt, h_gt)
        iou3d = iou2d * iouH
                               
        # Add a result
        n_corners = len(gt_floor_coor)

        if n_corners % 2 == 1:
            n_corners = 'odd'
        elif n_corners < 10:
            n_corners = str(n_corners)
        else:
            ##print('best 8',gt_path)
            n_corners = '10+'

        
        losses[n_corners]['2DIoU'].append(iou2d)
        losses[n_corners]['3DIoU'].append(iou3d)
        losses['overall']['2DIoU'].append(iou2d)
        losses['overall']['3DIoU'].append(iou3d)
    else:
        iou3d = 0


    return iou3d, iou2d, iouH

def get_z1(coory0, coory1, z0=50, coorH=512):
    v0 = np_coory2v(coory0, coorH)
    v1 = np_coory2v(coory1, coorH)
    c0 = z0 / np.tan(v0)
    z1 = c0 * np.tan(v1)
    return z1

def prepare_gtdt_pairs(gt_glob, dt_glob):
    
    gt_paths = sorted(glob.glob(gt_glob))

    dt_paths = dict([(os.path.split(v)[-1].split('.')[0], v)
                     for v in glob.glob(dt_glob) if v.endswith('json')])
        
    gtdt_pairs = []
    
    for gt_path in gt_paths:
        k = os.path.split(gt_path)[-1].split('.')[0]
        if k in dt_paths:
            gtdt_pairs.append((gt_path, dt_paths[k]))

    return gtdt_pairs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dt_glob', required=False, default = '../results/*json')
    
    parser.add_argument('--gt_glob', required=False, default = '../data/atlantalayout/test/label_cor/*')             
    parser.add_argument('--w', default=1024, type=int,
                        help='GT images width')
    parser.add_argument('--h', default=512, type=int,
                        help='GT images height')
    args = parser.parse_args()

    # Prepare (gt, dt) pairs        
    gtdt_pairs = prepare_gtdt_pairs(args.gt_glob, args.dt_glob)
        
    # Testing
    losses = dict([
        (n_corner, {'2DIoU': [], '3DIoU': []})
        for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
    ])

    idx = 0
        
    for gt_path, dt_path in tqdm(gtdt_pairs, desc='Testing'):
        # Parse ground truth
                
        idx = idx+1

        gt_cor_id = []
        with open(gt_path) as f:
            gt_cor_id = np.array([line.strip().split() for line in f if line.strip()], np.float32)
                
        # Parse inferenced result
        with open(dt_path) as f:
            dt = json.load(f)
        dt_cor_id = np.array(dt['uv'], np.float32)
        dt_cor_id[:, 0] *= args.w
        dt_cor_id[:, 1] *= args.h

        dt_z0 = np.array(dt['z0'], np.float32)
        dt_z1 = np.array(dt['z1'], np.float32)
               
        loc_iou3d, loc_iou2d, loc_iouH = test_shape(idx, dt_cor_id, dt_z0, dt_z1, gt_cor_id, args.w, args.h, losses)
               
                    
    for k, result in losses.items():
        iou2d = np.array(result['2DIoU'])
        iou3d = np.array(result['3DIoU'])
        if len(iou2d) == 0:
            continue
        print('GT #Corners: %s  (%d instances)' % (k, len(iou2d)))
        print('    2DIoU: %.2f' % (
            iou2d.mean() * 100,
        ))
        print('    3DIoU: %.2f' % (
            iou3d.mean() * 100,
        ))

