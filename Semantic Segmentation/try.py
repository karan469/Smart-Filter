import cv2
bg = cv2.imread('bg.jpeg')

font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2
   
position = (10,50)
cv2.putText(
     bg, #numpy array on which text is written
     "Python Examples", #text
     position, #position at which writing has to start
     cv2.FONT_HERSHEY_SIMPLEX, #font family
     700, #font size
     (209, 80, 0, 255), #font color
     3) #font stroke
   
# Displaying the image 
cv2.imwrite('bg_1.jpeg',bg)  
