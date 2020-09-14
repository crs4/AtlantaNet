#!/usr/bin/python3
import os
import sys  
import numpy as np
import json  as js

PI = float(np.pi)

def np_xy2coor(xy, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    xy: N x 2
    '''
    x = xy[:, 0] - floorW / 2 + 0.5
    y = xy[:, 1] - floorH / 2 + 0.5

    u = np.arctan2(x, -y)
    v = np.arctan(z / np.sqrt(x**2 + y**2))

    coorx = (u / (2 * PI) + 0.5) * coorW - 0.5
    coory = (-v / PI + 0.5) * coorH - 0.5

    return np.hstack([coorx[:, None], coory[:, None]])



if len(sys.argv)<5:
  print("Usage ",sys.argv[0]," json_dir output_dir width height")
  exit(1)

json_dir=sys.argv[1]
output_dir=sys.argv[2]
width=sys.argv[3]
height=sys.argv[4]

data_paths = [i for i in (os.path.join(json_dir, f) for f in os.listdir(json_dir)) if os.path.isfile(i)]

for json_file in data_paths:
  print("Processing ", json_file)  
  json_base = os.path.basename(json_file)
  scan_probe = os.path.splitext(json_base)[0]
  txt_file=output_dir+'/'+scan_probe+'.txt'
  with open(json_file, "r") as read_file, open(txt_file, "w") as write_file:
    data = js.load(read_file)
    #print(data)
    floorH=-data["cameraHeight"]
    ceilingH=data["layoutHeight"]-data["cameraHeight"]
    pixU = [] #np.zeros( data["layoutPoints"]["num"], 2)
    pixF = [] #np.zeros( data["layoutPoints"]["num"], 2)
    pixC = [] #np.zeros( data["layoutPoints"]["num"], 2)
    
    for p in data["layoutPoints"]["points"]:
      imgU = int(p["coords"][0]*float(width)-0.5)
      x =  p["xyz"][0]
      y =  p["xyz"][1]
      z =  p["xyz"][2]
      
      vfloor    = np.arctan(floorH/np.sqrt(x**2 + z**2))
      vceiling  = np.arctan(ceilingH/np.sqrt(x**2 + z**2))
      imgVfloor = int((-vfloor/PI+0.5)*float(height)-0.5)
      imgVceiling = int((-vceiling/PI+0.5)*float(height)-0.5)
                  
      print(imgU,imgVceiling,file=write_file)
      print(imgU,imgVfloor,file=write_file)
             
    read_file.close()
    write_file.close()
    
    




 

