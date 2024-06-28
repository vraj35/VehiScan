import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('veh2.mp4') # selecting the video


my_file = open("coco.txt", "r")  #splitting our classes line by line
data = my_file.read()
class_list = data.split("\n")  #cl represent coco file
#print(class_list)

count=0

tracker=Tracker()

cy1=322
cy2=368
offset=6

while True:    
    ret,frame = cap.read()  # for reading the video
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500)) #1020 width and 500 h
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float") #converts a to pandas data frame
#    print(px)
    list=[]
             
    for index,row in px.iterrows(): 
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]  # d 
        if 'car' in c:
            list.append([x1,y1,x2,y2]) #rectangle coordinates
    bbox_id=tracker.update(list)
    for bbox in bbox_id:  #center of the circle points
        x3,y3,x4,y4,id=bbox #new rectangle coordinates
        cx=int(x3+x4)//2   #center point for counting  
        cy=int(y3+y4)//2
        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
           



    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

