import os
import glob
##import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import math

import torch

from atlanta_model import AtlantaNet

from misc import tools, atlanta_transform, A2P, layout_viewer

###only for debug
import matplotlib.pyplot as plt
import cv2

#####dafault values
def_camera_h = 1.7 ####as a metric scale factor - camera height in meters
def_pth ='ckpt/resnet101_atlantalayout.pth' ##
def_output_dir = 'results/'

def_img = 'data/atlantalayout/test/img/2t7WUuJeko7_c2e11b94c07a4d6c85cc60286f586a02_equi.png' #

def cuda_to_cpu_tensor(x_tensors):
    x_tensors = x_tensors.cpu().numpy()
    sz = x_tensors.shape[0]
    x_imgs = []
    x_img = x_tensors[0 : sz]        
    x_imgs.append(x_img)
    return np.array(x_imgs)

def inference(net, x, device):
                           
    cont = net(x.to(device)) ### 
                 
    cont = cuda_to_cpu_tensor(cont.cpu()).mean(0)  
            
    return cont 

def h_from_contours(cp_prob, fp_prob):
    fp_prob_cont = tools.approx_shape(fp_prob)
    cp_prob_cont = tools.approx_shape(cp_prob)

    i_fp_prob = np.zeros(fp_prob.shape)
    i_cp_prob = np.zeros(cp_prob.shape)
        
    f_count = fp_prob_cont.shape[0]
       
    if(f_count>3):
        cv2.polylines(i_fp_prob, [fp_prob_cont], True, 255, 1)
    
    i_fp_prob = np.uint8(i_fp_prob)

    h_opt = 0.0

    h_max = 5.0

    max_i = 0
    
    for h in np.arange((def_camera_h+0.1), h_max, 0.05):
                      
       h_ratio =  (h - def_camera_h) / def_camera_h  
       
       cp_prob_scaled = cp_prob
       
       if(h_ratio>0 and fp_prob.shape[0]>0):
           cp_prob_scaled = tools.resize_crop(cp_prob, h_ratio, fp_prob.shape[0])

       cp_prob_cont = tools.approx_shape(cp_prob_scaled)
       
       i_cp_prob = np.zeros(cp_prob_scaled.shape)

       c_count = cp_prob_cont.shape[0]

       if(c_count>3):
           cv2.polylines(i_cp_prob, [cp_prob_cont], True, 255, 1)
       
       i_cp_prob = np.uint8(i_cp_prob)
                                               
       prob_i = cv2.bitwise_and(i_cp_prob,i_fp_prob)
               
       i_count = cv2.countNonZero(prob_i)
             
       if(i_count>max_i):
           max_i = i_count
           h_opt = h - def_camera_h
    
    if(h_opt>0):
        ceiling_height = h_opt
    else:
        ceiling_height = 1.3 ###default value is case of failure

    return ceiling_height    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=False, default = def_pth,
                        help='path to load saved checkpoint.')
    parser.add_argument('--img', required=False, default = def_img)
    parser.add_argument('--output_dir', required=False, default = def_output_dir)
    parser.add_argument('--visualize', action='store_true', default = True)
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    
    args = parser.parse_args()

    # Prepare image to processed
    paths = sorted(glob.glob(args.img))
    if len(paths) == 0:
        print('no images found')
    for path in paths:
        assert os.path.isfile(path), '%s not found' % path

    # Check target directory
    if not os.path.isdir(args.output_dir):
        print('Output directory %s not existed. Create one.' % args.output_dir)
        os.makedirs(args.output_dir)
    device = torch.device('cpu' if args.no_cuda else 'cuda')

    # Loaded trained model
    net = tools.load_trained_model(AtlantaNet, args.pth).to(device)
    net.eval()

    # Inferencing
    with torch.no_grad():
        for i_path in tqdm(paths, desc='Inferencing'):
            k = os.path.split(i_path)[-1][:-4]

            W = 1024
            H = 512

            # Load image
            img_pil = Image.open(i_path)

            if(len(img_pil.getbands())<3):
               img_pil = img_pil.convert("RGB")

            if img_pil.size != (W, H):
                img_pil = img_pil.resize((W, H), Image.BICUBIC)

            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            e_x = torch.FloatTensor([img_ori / 255])

            print('e_x shape',e_x.shape,'for image',i_path)                  
            
            e2p = A2P(out_dim=net.fp_size, gpu=False)

            [up_view, down_view] = e2p(e_x)
              
            x_up = torch.FloatTensor(up_view)
            x_down = torch.FloatTensor(down_view)                      
                                               
            # Inferecing shapes
            up_mask = inference(net, x_up, device)
            down_mask = inference(net, x_down, device)

            up_mask = up_mask.squeeze(0)
            down_mask = down_mask.squeeze(0)

            h_c_max = np.amax(up_mask)
            h_f_max = np.amax(down_mask)                        
                                   
                       
            up_mask_img = ( up_mask * 255/h_c_max ).astype(np.uint8)
            down_mask_img = ( down_mask * 255/h_f_max).astype(np.uint8)   
                                   
            cp_ret, cp_prob = cv2.threshold(up_mask_img, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            fp_ret, fp_prob = cv2.threshold(down_mask_img, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
            
            ##using cm to export
            h_f_mean = 100.0 * def_camera_h 
            h_c_mean = 100.0 * h_from_contours(cp_prob,fp_prob)

            print('Estimated heights:',h_c_mean,-h_f_mean)
            
                       
            scale_f = h_f_mean/atlanta_transform.fl
            scale_c = h_c_mean/atlanta_transform.fl
                                              
            cp_prob_metric = tools.resize(cp_prob, scale_c)
            fp_prob_metric = tools.resize(fp_prob, scale_f)
                                                            
            c_pts, r_c, c_area = tools.approx_shape(cp_prob_metric, return_reliability=True)
            f_pts, r_f, f_area = tools.approx_shape(fp_prob_metric, return_reliability=True)                       
                                             
            if( (r_c<0.7 and r_f>r_c) or len(c_pts)<3):
                ###ceiling dims unreliable using floor shape
                room_pts = f_pts
                scale = scale_f                    
            else:
                room_pts = c_pts  
                scale = scale_c
            
            ####recovering metric scale to save the model
            fp_size  = net.fp_size*scale               
            
            if(len(room_pts)>3):
                json_name = tools.export2json(room_pts, W, H, fp_size, args.output_dir, args.img, k, h_c_mean, -h_f_mean)                
            else:
                print('Failing to save model ',i_path)

#visualize output#####################                           
                                                          
            if(args.visualize):
                                              
                ### draw functions
                x_up_img = tools.x2image(x_up.squeeze(0))
                x_down_img = tools.x2image(x_down.squeeze(0))
                                
                footprint_up = x_up_img.copy()
                footprint_down = x_down_img.copy()

                footprint_up_metric = tools.resize(footprint_up, scale_c)
                footprint_down_metric = tools.resize(footprint_down, scale_f)

                if(len(c_pts)>0):
                    cv2.polylines(footprint_up_metric, [c_pts], True, (0,0,255),2,cv2.LINE_AA)             
               
                if(len(f_pts)>3):
                   cv2.polylines(footprint_down_metric, [f_pts], True, (255,0,0),2,cv2.LINE_AA)
            
                if (json_name is not None):
                    layout_viewer.show_3D_layout(args.img, json_name, def_camera_h)
                
                plt.figure(0)
                plt.title('Ceiling tensor with result')
                plt.imshow(footprint_up_metric)
                                
                plt.figure(1)
                plt.title('Floor tensor with result')
                plt.imshow(footprint_down_metric)
                                           
                plt.show()               
