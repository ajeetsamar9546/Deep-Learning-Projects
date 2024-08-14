from mtcnn import MTCNN
import cv2

detector =MTCNN()
img = cv2.imread('./Images/ajeet1.jpg')

output= detector.detect_faces(img)
print(output)


x,y,width,height=output[0]['box']

cv2.rectangle(img,pt1=(x,y),pt2=(x+width,y+height),color=(255,0,0))
cv2.imshow('window',img)
cv2.waitKey(0)
