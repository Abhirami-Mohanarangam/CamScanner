import cv2
import numpy as np 
import Corners

img = cv2.imread('sample.jpg')
img = cv2.resize(img,(1300,800))
org = img.copy()
cv2.imshow("Original Image",img)
cv2.waitKey()

gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_image = cv2.bilateralFilter(gray_image,11,17,17)


image = cv2.cvtColor(gray_image,cv2.COLOR_GRAY2RGB)

blurred_image = cv2.GaussianBlur(gray_image,(5,5),0)


canny_edge = cv2.Canny(gray_image,30,50)

contours,hierarchy = cv2.findContours(canny_edge.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key = cv2.contourArea, reverse = True)

x,y,w,h = cv2.boundingRect(contours[0])
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

for c in contours:
    p = 0.015 * cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,p,True)
    
    if len(approx) == 4:
        target = approx
        break
    
cv2.drawContours(img,[target], -1, (0,0,255), 2)


approx = Corners.getCorners(target)

coordinates = np.float32([[0,0],[800,0],[800,800],[0,800]])

crop = cv2.getPerspectiveTransform(approx,coordinates)
cropped_image = cv2.warpPerspective(org,crop,(800,800))
cv2.imshow("Snanned Image",cropped_image)
cv2.waitKey() 

