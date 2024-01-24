import os
from pascal_voc_writer import Writer as PascalWriter
import os
import sys
import cv2
import math
import json
import numpy as np
import shapely.geometry as shgeo

from PIL import Image
from tqdm import tqdm

small_count =0

def write_xml(obj,img_path,xml_name,img_width,img_height,results_dir):
    pascal_writer = PascalWriter(img_path, img_width, img_height)    
    for i in range(len(obj)):
        print(obj[i])
        print("class",obj[i]['name'])
        print("box",obj[i]['box'])
        pascal_writer.addObject(name=obj[i]['name'],
                                box=obj[i]['box'])
    pascal_writer.save(os.path.join(results_dir, xml_name))    
    
def parse_voc_poly(filename):
    objects = []
    #print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    content = f.read()
    content_splt = [x for x in content.split('<HRSC_Object>')[1:] if x!='']
    count = len(content_splt)
    if count > 0:
        for obj in content_splt:
            object_struct = {}
            object_struct['name'] = 'ship'
            object_struct['difficult'] = '0' 
            cx = float(obj[obj.find('<mbox_cx>')+9 : obj.find('</mbox_cx>')])
            cy = float(obj[obj.find('<mbox_cy>')+9 : obj.find('</mbox_cy>')])
            w  = float(obj[obj.find('<mbox_w>')+8 : obj.find('</mbox_w>')])
            h  = float(obj[obj.find('<mbox_h>')+8 : obj.find('</mbox_h>')])
            a  = obj[obj.find('<mbox_ang>')+10 : obj.find('</mbox_ang>')]
            a = float(a) if not a[0]=='-' else -float(a[1:])
            if a < 0:
                theta = a +math.pi
            else:
                theta = a
            #points = cv2.boxPoints(((cx,cy),(w,h),theta))
            object_struct['box'] =[cx,cy,w,h,theta]
            #object_struct['poly'] = [points[0],points[1],points[2],points[3]]
            #gtpoly = shgeo.Polygon(object_struct['poly'])
            #object_struct['area'] = gtpoly.area
            #poly = list(map(lambda x:np.array(x), object_struct['poly']))
            #object_struct['long-axis'] =1# max(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            #object_struct['short-axis'] =1# min(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            #if (object_struct['long-axis'] < 15):
            #    object_struct['difficult'] = '1'
            #    global small_count
            #    small_count = small_count + 1
            objects.append(object_struct)
    else:
        print('No object founded in %s' % filename)
    return objects    

def hrscTolabelme(ori_path,ori_img_path,det_path):
    if not os.path.exists(det_path):
        os.makedirs(det_path)
        
    for xmlName in tqdm(os.listdir(ori_path)):
        #print(xmlName)
        tmp_name = xmlName.split('.', 1)[0]
        #print(tmp_name)
        xml_path = ori_path + xmlName
        print(xml_path)
        img_path = ori_img_path + tmp_name +'.jpg'
        img = Image.open(img_path)
        img_height = img.height
        img_width = img.width
        print(img_path,img_width,img_height)
        obj = parse_voc_poly(xml_path)
        write_xml(obj,img_path,xmlName,img_width,img_height,det_path)
        #write_xml(obj,img_name,xml_name,img_height,results_dir)
    #write_xml(obj,TEST_RESULTS_DIR)
    #print(obj)
    
    
def main():
    ori_path = "/mnt/dataset/HRSC2016/Test/Annotations/"
    ori_img_path = "/mnt/dataset/HRSC2016/Test/AllImages/"
    det_path = "/mnt/dataset/HRSC2016/Test/roilabelme/"
    hrscTolabelme(ori_path,ori_img_path,det_path)


if __name__ == '__main__':
   main()    