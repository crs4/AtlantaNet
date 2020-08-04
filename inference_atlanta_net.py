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

def_camera_h = 1.7 ####as a metric scale factor

def_pth ='ckpt/matterport_layout.pth' ## best 227 temp 96 temp 30

def_output_dir = 'results/matterportlayout/'
##def_output_dir = 'results/'

##def_img = 'C:/vic/trunk/software/atlanta_net/data/matterport3D_test_clean/img/*.*'
def_img = 'C:/vic/trunk/software/atlanta_net/data/matterport3D_test_clean/img/7y3sRwLe3Va_1410b021e1c14f529188eb026fbb369a.png' ### from M3D best test - Adam perfect

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
               ##break
    
    return h_opt    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=False, default = def_pth,
                        help='path to load saved checkpoint.')
    parser.add_argument('--img_glob', required=False, default = def_img)
    parser.add_argument('--output_dir', required=False, default = def_output_dir)
    parser.add_argument('--visualize', action='store_true', default = True)
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    
    args = parser.parse_args()

    # Prepare image to processed
    paths = sorted(glob.glob(args.img_glob))
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
            
            h_f_mean = def_camera_h*100.0
            h_c_mean = 130.0 ###default - backup in case h form contour fails

            ###NB. fixed focal is h:90.2
            
            h_c = 100.0*h_from_contours(cp_prob,fp_prob)
            
            if h_c>0:
                h_c_mean = h_c
              
                        
            scale_f = h_f_mean/atlanta_transform.fl
            scale_c = h_c_mean/atlanta_transform.fl

            ##print('floor metric scale',scale_f, scale_c)
                       
            cp_prob_metric = tools.resize(cp_prob, scale_c)
            fp_prob_metric = tools.resize(fp_prob, scale_f)
                                                            
            c_pts, r_c, c_area = tools.approx_shape(cp_prob_metric, return_reliability=True)
            f_pts, r_f, f_area = tools.approx_shape(fp_prob_metric, return_reliability=True)                       

            ##print('ceiling reliability and area',r_c, c_area,'floor reliability and area',r_f, f_area)  
            
            area_ratio = min(c_area,f_area) / max(c_area,f_area)

            ##print('area ratio',area_ratio)
                      
            if( (r_c<0.7 and r_f>r_c)):##or (area_ratio<0.25) ):
                ###ceiling dims unreliable using floor shape
                room_pts = f_pts
                scale = scale_f                    
            else:
                room_pts = c_pts  
                scale = scale_c
            
            fp_size  = net.fp_size*scale
            
            print('saving heights',h_c_mean,-h_f_mean)
            
            if(len(room_pts)>3):
                ##rint('2D shape', room_pts.shape)
                json_name = tools.export2json(room_pts, W, H, fp_size, args.output_dir, def_img, k, h_c_mean, -h_f_mean)
                
            else:
                print('height recovery failed for',i_path)

##################################################################################################################################                                  
                                                          
            if(args.visualize):
                output_seg = False

                if(output_seg):
                    bon = tools.transform2equi(room_pts,h_c_mean,h_f_mean, W, H, net.fp_size, scale)           
                                                                                                                                               
                    vis_out = tools.visualize_equi_model(e_x.squeeze(), torch.FloatTensor(bon.copy()))

                    if (vis_out is not None):
                        plt.figure(6)
                        plt.title('Equi mask')
                        plt.imshow(vis_out)

                    vis_path = os.path.join(args.output_dir, k + '.raw.png')
                    vh, vw = vis_out.shape[:2]
                    Image.fromarray(vis_out).save(vis_path)


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
                    layout_viewer.show_3D_layout(def_img, json_name, def_camera_h)
                
                plt.figure(0)
                plt.title('Ceiling tensor with result')
                plt.imshow(footprint_up_metric)

                plt.figure(1)
                plt.title('Inferred ceiling mask')
                plt.imshow(up_mask_img)

                plt.figure(2)
                plt.title('Filtered ceiling mask')
                plt.imshow(cp_prob)

                plt.figure(3)
                plt.title('Floor tensor with result')
                plt.imshow(footprint_down_metric)

                plt.figure(4)
                plt.title('Inferred floor mask')
                plt.imshow(down_mask_img)

                plt.figure(5)
                plt.title('Filtered floor mask')
                plt.imshow(fp_prob)
                                           
                plt.show()                   
            
                      
            

   
