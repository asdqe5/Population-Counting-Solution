#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:51:33 2017

@author: kyleguan
"""
import numpy as np
import cv2 #opencv는 오픈 소스 컴퓨터 비전, 영상처리 라이브러리이다. 객체 얼굴 행동인식 모션 추적등의 응용프로그램에서 사용한다.

class Box: #box 클래스를 정의한다.
    def __init__(self): #클래스 생성자 이용 -> 객체가 생성될 때 자동으로 호출되는 메서드를 의미한다.
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float() #Box클래스로 생성되는 객체의 변수들을 선언한다.

def overlap(x1,w1,x2,w2): #x1 ,x2 are two box center x, 교차하는 box의 x좌표와 너비 또는 y좌표와 높이가 들어왔을 때 교차하는 부분의 너비 또는 높이를 리턴한다.
    l1 = x1 - w1 / 2.; #파이썬에서는 세미콜론을 붙이나 떼나 똑같다.
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b): #교차하는 box의 너비와 높이를 구하여 넓이를 리턴한다.
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):#교차하는 box 전체의 넓이를 리턴한다. a,b box넓이의 합에 교차하는 부분의 넓이를 뺀다. 
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):#사용하는 object detector의 능력을 평가하기 위해서 intersection of union을 계산한다. iou가 클수록 잘 detection한 것!
    return box_intersection(a, b) / box_union(a, b);



def box_iou2(a, b):#사용하는 object detector의 능력을 평가하기 위해서 intersection of union을 계산한다. iou가 클수록 잘 detection한 것!
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''
    
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))#intersection의 너비
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))#intersection의 높이
    s_intsec = w_intsec * h_intsec #intersesction의 넓이
    s_a = (a[2] - a[0])*(a[3] - a[1])#a box의 넓이
    s_b = (b[2] - b[0])*(b[3] - b[1])# b box의 넓이
  
    return float(s_intsec)/(s_a + s_b -s_intsec) #iou 계산



def convert_to_pixel(box_yolo, img, crop_range):#bounding box의 크기 좌표값을 pixel단위 좌표값으로 변환한다.
    '''
    Helper function to convert (scaled) coordinates of a bounding box 
    to pixel coordinates. 
    
    Example (0.89361443264143803, 0.4880486045564924, 0.23544462956491041, 
    0.36866588651069609)
    
    crop_range: specifies the part of image to be cropped #자를 이미지를 지정한다.
    '''
    
    box = box_yolo
    imgcv = img
    [xmin, xmax] = crop_range[0]
    [ymin, ymax] = crop_range[1]
    h, w, _ = imgcv.shape
    
    # Calculate left, top, width, and height of the bounding box, box의 x,y,w,h값은 비율로 들어온다.
    left = int((box.x - box.w/2.)*(xmax - xmin) + xmin)
    #left = int((box.x - box.w/2)*xmin)
    top = int((box.y - box.h/2.)*(ymax - ymin) + ymin)
    
    width = int(box.w*(xmax - xmin))
    height = int(box.h*(ymax - ymin))
    
    # Deal with corner cases, box가 코너일 경우를 다뤄준다.
    if left  < 0    :  left = 0
    if top   < 0    :   top = 0
    
    # Return the coordinates (in the unit of the pixels)
  
    box_pixel = np.array([left, top, width, height])#넘파이 어레이로 변환
    return box_pixel



def convert_to_cv2bbox(bbox, img_dim = (1280, 720)):
    '''
    Helper fucntion for converting bbox to bbox_cv2
    bbox = [left, top, width, height]
    bbox_cv2 = [left, top, right, bottom]
    img_dim: dimension of the image, img_dim[0]<-> x
    img_dim[1]<-> y
    '''
    left = np.maximum(0, bbox[0])
    top = np.maximum(0, bbox[1])
    right = np.minimum(img_dim[0], bbox[0] + bbox[2])
    bottom = np.minimum(img_dim[1], bbox[1] + bbox[3])
    
    return (left, top, right, bottom)
    
    
def draw_box_label(id, img, bbox_cv2, box_color=(0, 255, 255), show_label=True):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX #글꼴 설정, 보통 크기의 세리프 글꼴
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2] #bbox_cv2 np 변수
    
    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 4) #좌측 상단 모서리와 우측 하단 모서리가 연결된(B,G,R)색상, 두께 굵기의 사각형을 그린다.
    
    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        cv2.rectangle(img, (left-2, top-45), (right+2, top), box_color, -1, 1) #박스를 채워 그린다.
        
        # Output the labels that show the x and y coordinates of the bounding box center.
        text_x= 'id='+str(id) #text_x는 id값
        cv2.putText(img,text_x,(left,top-25), font, font_size, font_color, 1, cv2.LINE_AA)#cv2.putText(이미지, 문자, (x, y), 글꼴, 글자 크기, (B, G, R), 두께, 선형 타입)
        text_y= 'y='+str((top+bottom)/2) #bounding box의 중앙 y 좌표
        cv2.putText(img,text_y,(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)
    
    return img

def draw_centroid(id, img, bbox_cv2, box_color = (0, 255, 255), show_label = True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]
    center_x = int((left + right)/2.0)
    center_y = int((top + bottom)/2.0)
    #Draw the centroid
    cv2.circle(img, (center_x, center_y), 4, (0, 255, 0), -1)
    text = "ID {}".format(id)
    cv2.putText(img, text, (center_x - 10, center_y - 10), font, font_size, font_color, 1, cv2.LINE_AA)

    return img