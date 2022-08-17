import os
import cv2
from paddleocr import PaddleOCR,draw_ocr
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image, ImageDraw
from PIL import ImagePath
import numpy as np
import os
import pandas as pd
import json
import tensorflow as tf
import pytesseract
import shutil

dirpath = os.path.join('temp')
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)
os.mkdir("./temp")

model = tf.keras.models.load_model('./saved_model')

# given predicted boxes approximate the predicted rectangles
def fil_approx_boxes(img):
    cv2.imwrite("temp/test.jpeg",img)
    img = cv2.imread("temp/test.jpeg",0)
    img = cv2.medianBlur(img,5)
    img = cv2.GaussianBlur(img,(13,13),0)
    img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]

    _, threshold = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if x==0 or y==0:
            continue
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),-1)

    img = cv2.GaussianBlur(img,(13,13),0)
    img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]
    return img

def intersection(box1,box2):
    return [box2[0],box1[1],box2[2],box1[3]]

def iou(box1,box2):
    x1=max(box1[0],box2[0])
    y1=max(box1[1],box2[1])
    x2=min(box1[2],box2[2])
    y2=min(box1[3],box2[3])
    inter=abs(max((x2-x1,0))*max((y2-y1),0))
    if inter==0:
        return 0
    box1area=abs((box1[2]-box1[0])*(box1[3]-box1[1]))
    box2area=abs((box2[2]-box2[0])*(box2[3]-box2[1]))
    return inter/float(box1area+box2area-inter)

#  Given masked image, Save both tables and extract text from each
def extract_text1(img_path="temp/final_masked.png"):
    ocr=PaddleOCR(lang="en")
    image_path=img_path
    image_cv=cv2.imread(image_path)
    image_height=image_cv.shape[0]
    image_width=image_cv.shape[1]
    output=ocr.ocr(image_path)
    boxes=[line[0] for line in output]
    texts=[line[1][0] for line in output]
    probabilities=[line[1][1] for line in output]
    image_boxes=image_cv.copy()
    for box,text in zip(boxes,texts):
      cv2.rectangle(image_boxes,(int(box[0][0]),int(box[0][1])),(int(box[2][0]),int(box[2][1])),(0,0,255),1)
      cv2.putText(image_boxes,text,(int(box[0][0]),int(box[0][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(222,0,0),1)
    im=image_cv.copy()
    horiz_boxes=[]
    vert_boxes=[]

    for box in boxes:
        x_h,x_v=0,int(box[0][0])
        y_h,y_v=int(box[0][1]),0
        width_h,width_v=image_width,int(box[2][0]-box[0][0])
        height_h,height_v=int(box[2][1]-box[0][1]),image_height

        horiz_boxes.append([x_h,y_h,x_h+width_h,y_h+height_h])
        vert_boxes.append([x_v,y_v,x_v+width_v,y_v+height_v])

        cv2.rectangle(im,(x_h,y_h),(x_h+width_h,y_h+height_h),(255,255,0),1)
        cv2.rectangle(im,(x_v,y_v),(x_v+width_v,y_v+height_v),(255,255,0),1)
    import tensorflow as tf
    horiz_out=tf.image.non_max_suppression(horiz_boxes,
                                       probabilities,
                                       max_output_size=1000,
                                       iou_threshold=0.1,
                                       score_threshold=float('-inf'),
                                       name=None)
    import numpy as np
    horiz_lines=np.sort(np.array(horiz_out))
    in_nms=image_cv.copy()

    for val in horiz_lines:
      cv2.rectangle(in_nms,(int(horiz_boxes[val][0]),int(horiz_boxes[val][1])),(int(horiz_boxes[val][2]),int(horiz_boxes[val][3])),(0,0,255),1)
    vert_out=tf.image.non_max_suppression(vert_boxes,
                                      probabilities,
                                      max_output_size=1000,
                                      iou_threshold=0.1,
                                      score_threshold=float('-inf'),
                                      name=None)

    vert_lines=np.sort(np.array(vert_out))


    for val in vert_lines:
        cv2.rectangle(in_nms,(int(vert_boxes[val][0]),int(vert_boxes[val][1])),(int(vert_boxes[val][2]),int(vert_boxes[val][3])),(0,0,255),1)
    out_array=[["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]

    unordered_boxes=[]

    for i in vert_lines:
        unordered_boxes.append(vert_boxes[i][0])

    ordered_boxes=np.argsort(unordered_boxes)

    for i in range(len(horiz_lines)):
      for j in range(len(vert_lines)):
        resultant=intersection(horiz_boxes[horiz_lines[i]],vert_boxes[vert_lines[ordered_boxes[j]]])

        for b in range(len(boxes)):
            the_box=[boxes[b][0][0],boxes[b][0][1],boxes[b][2][0],boxes[b][2][1]]
            if (iou(resultant,the_box)>0.1):
                out_array[i][j]=texts[b]

    import pandas as pd
    df=pd.DataFrame(out_array)
    return df

# predict table and column masks and display
def predict_table_masks(img):
  res1, res2 = model.predict(np.array([img]))
  res1 =  np.expand_dims(np.argmax(res1[0], axis=-1), axis=-1)
  res2 = np.expand_dims(np.argmax(res2[0], axis=-1), axis=-1)
  pred_col = np.squeeze(np.where(res1==1,255,0))
  pred_table = np.squeeze(np.where(res2==1,255,0))
  return fil_approx_boxes(pred_table),fil_approx_boxes(pred_col)


#  Given masked image, Save both tables and extract text from each
# Predict masks and extract text
def predict_and_extract(img_path):
  image = tf.io.read_file(img_path)
  org_image = tf.image.decode_image(image, channels=3)
  h,w = org_image.shape[0],org_image.shape[1]

  image = tf.image.resize(org_image, [800, 800])
  pred_table, pred_col = predict_table_masks(image)
  tab = np.where(pred_table == 0,0,1)
  mask = np.expand_dims(tab,axis=2)
  mask = np.concatenate((mask,mask,mask),axis=2)
  cv2.imwrite("temp/mask.png",mask)

  mask = cv2.resize(cv2.imread("temp/mask.png"), (w,h), interpolation = cv2.INTER_AREA)
  masked_img= org_image.numpy() * mask
  cv2.imwrite("temp/final_masked.png",masked_img)
  return extract_text1()