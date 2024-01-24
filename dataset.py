# -*- coding: utf-8 -*-
import os
import cv2
import math
import random
import numpy as np
import torch.utils.data as data
import pycocotools.coco as coco
import utility_lite


class ctDataset(data.Dataset):
    num_classes = utility_lite.data_cls
    default_resolution = utility_lite.data_res#[512,512]
    mean = utility_lite.data_mean #np.array([0.5194416012442385,0.5378052387430711,0.533462090585746], dtype=np.float32).reshape(1, 1, 3)
    std  = utility_lite.data_std#np.array([0.3001546018824507, 0.28620901391179554, 0.3014112676161966], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, data_dir=utility_lite.data_dir, split='train'):#./data
        self.data_dir = os.path.join(data_dir, utility_lite.data_name)#crack50
        self.img_dir = os.path.join(self.data_dir, 'images')
        try:
            if split == 'train':
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'train.json')
            elif split == 'val':
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'val.json')
        except:
            print('No any data!')

        self.max_objs = 128
        self.class_name = ['obj']
        self._valid_ids = [1]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],dtype=np.float32)
        self._eig_vec = np.array([
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)

        self.split = split
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids) 
        num_objs = min(len(anns), self.max_objs)
        if utility_lite.DataMod ==1:
            img_path = img_path+ ".jpg"
        #print("img_path",img_path)
        if utility_lite.DataMod == 0:
            img_path = img_path.replace('.jpg',utility_lite.postfix1)
        #print(img_path)
        img = cv2.imread(img_path)  
        height, width = img.shape[0], img.shape[1]  
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  
    
        keep_res = False#False #
        if keep_res:
            input_h = (height | 31) + 1   
            input_w = (width | 31) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0  
            input_h, input_w = utility_lite.data_res #512, 512 

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,(input_w, input_h),flags=cv2.INTER_LINEAR) 
        inp = (inp.astype(np.float32) / 255.)  

        #��һ��
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1) 
        
        down_ratio = 4 
        output_h = input_h // down_ratio
        output_w = input_w // down_ratio
        num_classes = self.num_classes 
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])  

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)   
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        ang = np.zeros((self.max_objs, 1), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32) 
        ind = np.zeros((self.max_objs), dtype=np.int64) 
        
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8) 
        draw_gaussian = draw_umich_gaussian
        for k in range(num_objs):  
            ann = anns[k]  
            bbox, an = coco_box_to_bbox(ann['bbox']) 
            cls_id = int(self.cat_ids[ann['category_id']]) 
            bbox[:2] = affine_transform(bbox[:2], trans_output)    
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)  
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            # ���漸�ж��������������resize֮��ı任������Ҫ
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                if utility_lite.Heat_map == "OriAware":
                    radius_x = gaussian_radius((math.ceil(w), math.ceil(w)))
                    radius_y = gaussian_radius((math.ceil(h), math.ceil(h)))
                    #print("radius_x",radius_x)
                    #print("radius_y",radius_y)
                    radius_x = max(0, int(radius_x))
                    radius_y = max(0, int(radius_y))
                    #ct = np.array([cen_x, cen_y], dtype=np.float32)
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32) 
                    ct_int = ct.astype(np.int32)
                    #print("before:",ct_int)
                    #draw_msra_gaussian_rot(hm[cls_id], ct_int, radius_x,radius_y,an*math.pi/180)
                    #print("after:",ct_int)
                else:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))  
                    radius = max(0, int(radius))
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32) 
                    ct_int = ct.astype(np.int32) 
                    draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ang[k] = 1. * an
                ind[k] = ct_int[1] * output_w + ct_int[0]  
                reg[k] = ct - ct_int
                reg_mask[k] = 1
        #print("img_path",self.img_dir, file_name)
        if utility_lite.Heat_map == "OriAware":
          npy_name = file_name.split(".", 1)[0]        
          npy_save_path =self.img_dir + '/'+ str(utility_lite.data_res[1]) + 'HMmse' + npy_name  +'.npy' #save_path + 'HMmse' + npy_name  +'.npy'
          #print(npy_save_path)
          hm = np.load(npy_save_path)
        #cheak heat map
        #while(1):True        
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'ang':ang}
        reg_offset_flag = True #
        if reg_offset_flag:
            ret.update({'reg': reg})
        return ret
    
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)
    
def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2
    
def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])
    
def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha
    
def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)
    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)
    
def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result    
    
def get_affine_transform(center,scale,rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32),inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian2D_Rot(shape, sigma_x=1,sigma_y=1,theta=0):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    cov = [[sigma_y** 2, 0], [0, sigma_x**2 ]]   
    rot_matrix = np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) 
    #print("rot_matrix",rot_matrix)
    rot_cov = np.matmul(rot_matrix.T ,cov)
    rot_cov = np.matmul(rot_cov ,rot_matrix)
    #print(x.shape,y.shape)
    #print(len(x[0]),len(y[:;]),x,y)
    h = np.zeros((len(x[0]),len(y)))
    for i in range(len(x[0])):
        x_tmp = x[0][i]
        #print(x_tmp)
        for j in range(len(y)):
            y_tmp = y[j][0]              
            #print(y_tmp)            
            res = - ( np.matmul(np.matmul(np.matrix([x_tmp - x0,y_tmp - y0]),np.linalg.pinv(rot_cov)),np.matrix([x_tmp - x0,y_tmp - y0]).T))
            #res = - ( np.matmul(np.matmul(np.matrix([x_tmp - x0,y_tmp - y0]),np.matrix(rot_cov).I),np.matrix([x_tmp - x0,y_tmp - y0]).T))
            res = np.array(res*0.5)
            #print(res)
            h[i][j] = np.exp(res[0][0])    
    #h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h = np.exp(-(x * x + y * y) / (2 * sigma_x * sigma_x))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h   

def draw_umich_gaussian_rot(heatmap, center,sigma_x,sigma_y, theta= 0.5*np.pi, k=1):
    theta = theta -0.5*np.pi
    radius = max(sigma_x,sigma_y)
    diameter = 2 * radius + 1
    #gaussian = gaussian2D((diameter, diameter), diameter / 6)
    #print(gaussian)
    gaussian = gaussian2D_Rot((diameter, diameter), sigma_x / 6,sigma_y/6,theta)
  
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1) 
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right] 
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right] 
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_msra_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap

def draw_msra_gaussian_rot(heatmap, center, sigma_x,sigma_y,theta=1):
    theta = theta -0.5*np.pi
    sigma = max(sigma_x,sigma_y)
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    cov = [[sigma_y** 2, 0], [0, sigma_x**2 ]]   
    rot_matrix = np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) 
    #print("rot_matrix",rot_matrix)
    rot_cov = np.matmul(rot_matrix.T ,cov)
    rot_cov = np.matmul(rot_cov ,rot_matrix)
    #print("cov",cov)
    #print(len(x),len(y),rot_cov)
    #print(np.matrix(rot_cov).I)
    g = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        x_tmp = x[i]
        #print(x_tmp)
        for j in range(len(y)):
          y_tmp = y[j]              
          #print(y_tmp)
          res = - ( np.matmul(np.matmul(np.matrix([x_tmp - x0,y_tmp - y0]),np.linalg.pinv(rot_cov)),np.matrix([x_tmp - x0,y_tmp - y0]).T))  
          #res = - ( np.matmul(np.matmul(np.matrix([x_tmp - x0,y_tmp - y0]),np.matrix(rot_cov).I),np.matrix([x_tmp - x0,y_tmp - y0]).T))
          res = np.array(res*0.5) 
          g[i][j] = np.exp(res[0][0])
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1) 
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right] 
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right] 
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
    
def coco_box_to_bbox(box):
    bbox = np.array([box[0] - box[2]/2, box[1] - box[3]/2, box[0] + box[2]/2, box[1] + box[3]/2],dtype=np.float32)
    ang = float(box[4])
    return bbox, ang

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter*2+1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)
  
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                             radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                      radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i




