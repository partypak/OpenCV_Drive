# 출처 : https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
# 라이선스 : 학습으로만 사용.

import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("./yolo_object_detection/yolov3.weights", "./yolo_object_detection/yolov3.cfg")
classes = []
with open("./yolo_object_detection/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
##template  사진들 설정
cap = cv2.VideoCapture('./data/abc.mp4') # 0번 카메라
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

green = cv2.imread('./data/tr/green.jpg')
green= cv2.resize(green,(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
red = cv2.imread('./data/tr/red.jpg')
red= cv2.resize(red,(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

##Template img
temg1 = cv2.imread('./data/tr/g1.jpg')
temg2 = cv2.imread('./data/tr/g2.jpg')
temg3 = cv2.imread('./data/tr/g3.jpg')
temg4 = cv2.imread('./data/tr/g5.jpg')
temg5 = cv2.imread('./data/tr/g6.jpg')



temr1 = cv2.imread('./data/tr/r5.jpg')
temr2 = cv2.imread('./data/tr/r2.jpg')





temn1 = cv2.imread('./data/tr/n2.jpg')
temn2 = cv2.imread('./data/tr/n3.jpg')

th, tw = temg1.shape[:-1]
thg,twg = temg4.shape[:-1]
th2, tw2 = temr1.shape[:-1]
th3, tw3 = temn1.shape[:-1]


out1 = cv2.VideoWriter('./data/recorda.mp4',fourcc, 20.0, frame_size)
textGo = 'Go'
textStop = 'Stop'
textNum='50'
org = (450,450)
org2 = (500,450)
org3 = (600,450)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Loading image
    retval, img = cap.read()
    if not retval:
        break   
    ##초록불 감지
    resultg1 = cv2.matchTemplate(img, temg1 , cv2.TM_CCOEFF_NORMED)
    locg1 = np.where(resultg1 >= 0.65)
    resultg2 = cv2.matchTemplate(img, temg2 , cv2.TM_CCOEFF_NORMED)
    locg2 = np.where(resultg2 >= 0.65)
    resultg3 = cv2.matchTemplate(img, temg3 , cv2.TM_CCOEFF_NORMED)
    locg3 = np.where(resultg3 >= 0.65)
    resultg4 = cv2.matchTemplate(img, temg4 , cv2.TM_CCOEFF_NORMED)
    locg4 = np.where(resultg4 >= 0.65)
    resultg5 = cv2.matchTemplate(img, temg5 , cv2.TM_CCOEFF_NORMED)
    locg5 = np.where(resultg5 >= 0.65)
    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(resultg1)
    top_left = maxLoc
    match_val = maxVal
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    
    minValg2, maxValg2, minLocg2, maxLocg2 = cv2.minMaxLoc(resultg2)
    top_leftg2 = maxLocg2
    match_valg2 = maxValg2
    bottom_rightg2 = (top_leftg2[0] + tw, top_leftg2[1] + th)
    
    minValg3, maxValg3, minLocg3, maxLocg3 = cv2.minMaxLoc(resultg3)
    top_leftg3 = maxLocg3
    match_valg3 = maxValg3
    bottom_rightg3 = (top_leftg3[0] + tw, top_leftg3[1] + th)
    
    minValg4, maxValg4, minLocg4, maxLocg4 = cv2.minMaxLoc(resultg4)
    top_leftg4 = maxLocg4
    match_valg4 = maxValg4
    bottom_rightg4 = (top_leftg4[0] + twg, top_leftg4[1] + thg)
    
    minValg5, maxValg5, minLocg5, maxLocg5 = cv2.minMaxLoc(resultg5)
    top_leftg5 = maxLocg5
    match_valg5 = maxValg5
    bottom_rightg5 = (top_leftg5[0] + twg, top_leftg5[1] + thg)
    
     ##빨간불 감지
    resultr1 = cv2.matchTemplate(img, temr1,cv2.TM_CCOEFF_NORMED)
    locr1 = np.where(resultr1 >= 0.7)
    resultr2 = cv2.matchTemplate(img, temr2,cv2.TM_CCOEFF_NORMED)
    locr2 = np.where(resultr2 >= 0.7)
  
    
    
    
    minValr1, maxValr1, minLocr1, maxLocr1 = cv2.minMaxLoc(resultr1)
    top_leftr1 = maxLocr1
    match_valr1 = maxValr1
    bottom_rightr1 = (top_leftr1[0] + tw3, top_leftr1[1] + th3)
    
    minValr2, maxValr2, minLocr2, maxLocr2 = cv2.minMaxLoc(resultr2)
    top_leftr2 = maxLocr2
    match_valr2 = maxValr2
    bottom_rightr2 = (top_leftr2[0] + tw3, top_leftr2[1] + th3)
    
    
    
    
     ##속도제한 감지
    resultn1 = cv2.matchTemplate(img, temn1,cv2.TM_CCOEFF_NORMED)
    locn1 = np.where(resultn1 >= 0.6)
  
    resultn2 = cv2.matchTemplate(img, temn2,cv2.TM_CCOEFF_NORMED)
    locn2 = np.where(resultn2 >= 0.6)
    
    minValn1, maxValn1, minLocn1, maxLocn1 = cv2.minMaxLoc(resultn1)
    top_leftn1 = maxLocn1
    match_valn1 = maxValn1
    bottom_rightn1 = (top_leftn1[0] + tw2, top_leftn1[1] + th2)
   
    minValn2, maxValn2, minLocn2, maxLocn2 = cv2.minMaxLoc(resultn2)
    top_leftn2 = maxLocn2
    match_valn2 = maxValn2
    bottom_rightn2 = (top_leftn2[0] + tw2, top_leftn2[1] + th2)
  
    ##초록불 감지
    for top_left in zip(*locg1[::-1]):
       
        cv2.rectangle(img, top_left, (top_left[0] + tw, top_left[1] + th), (0, 255, 0), 3)
        cv2.putText(img,textGo, org, font, 5, (0,255,0), 3)
    for top_leftg2 in zip(*locg2[::-1]):
       
        cv2.rectangle(img, top_leftg2, (top_leftg2[0] + tw, top_leftg2[1] + th), (0, 255, 0), 3)
        cv2.putText(img,textGo, org, font, 5, (0,255,0), 3)    
    for top_leftg3 in zip(*locg3[::-1]):
       
        cv2.rectangle(img, top_leftg3, (top_leftg3[0] + tw, top_leftg3[1] + th), (0, 255, 0), 3)
        cv2.putText(img,textGo, org, font, 5, (0,255,0), 3) 
    for top_leftg5 in zip(*locg5[::-1]):
       
        cv2.rectangle(img, top_leftg5, (top_leftg5[0] + tw, top_leftg5[1] + th), (0, 255, 0), 3)
        cv2.putText(img,textGo, org, font, 5, (0,255,0), 3) 
                    
        
        
        
        
        
    ##빨간불    
        
    for top_leftr1 in zip(*locr1[::-1]): 
        img=cv2.add(red,img)   
        
        cv2.rectangle(img, top_leftr1,(top_leftr1[0] + tw2, top_leftr1[1] + th2), (255, 255, 255), 3)
        cv2.putText(img,textStop, org2, font, 5, (0,0,255), 3)
    for top_leftr2 in zip(*locr2[::-1]): 
        img=cv2.add(red,img)   
       
        cv2.rectangle(img, top_leftr2,(top_leftr2[0] + tw2, top_leftr2[1] + th2), (255, 255, 255), 3)
        cv2.putText(img,textStop, org2, font, 5, (0,0,255), 3)
        
    
    
    ##속도제한 감지
    
    
    for top_leftn1 in zip(*locn1[::-1]): 
        
       
        cv2.rectangle(img, top_leftn1,(top_leftn1[0] + tw3, top_leftn1[1] + th3), (0, 0, 255), 3)
        cv2.putText(img,textNum, org3, font, 5, (0,0,255), 3)    
    for top_leftn2 in zip(*locn2[::-1]): 
          
       
        cv2.rectangle(img, top_leftn2,(top_leftn2[0] + tw3, top_leftn2[1] + th3), (0, 0, 255), 3)
        cv2.putText(img,textNum, org3, font, 5, (0,0,255), 3)  

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
           ## cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    key = cv2.waitKey(25)
    if key == 27:
        break
    out1.write(img)
    cv2.imshow("Image", img)
    # cv2.waitKey(1)
    

cap.release()
out1.release()