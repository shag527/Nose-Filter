import cv2

face_cascsde=cv2.CascadeClassifier(r'C:\Users\hp\PycharmProjects\opencv work\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
nose=cv2.imread('pignose.png')
nose=cv2.cvtColor(nose,cv2.COLOR_BGR2GRAY)


while True:
    ret,img=cap.read()
    cap.set(3,200)
    cap.set(4,400)
    if ret:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascsde.detectMultiScale(img,1.1,4)
        for (x,y,w,h) in faces:
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            #cv2.line(img,(x,y+h),(x+w,y+h),(0,255,0),4)
            #cv2.circle(img,(x+int(w/2),y+int(h/2)),4,(0,0,255),-1)
            #cv2.line(img,(x-int(w/10)+int(w/2),y+int(h/2)),(x+int(w/10)+int(w/2),y+int(h/2)),(0,255,0),4)
            nose_width=2*int(w/10)
            nose_height=nose_width
            nose=cv2.resize(nose,(nose_width,nose_height))
            nose_center=(x+int(w/2),y+int(h/2))
            nose_top_left=(x+int(w/2)-int(nose_width/2),y+int(h/2)-int(nose_height/2)+10)
            nose_bottom_right=(x+int(w/2)+int(nose_width/2),y+int(h/2)+int(nose_height/2)+10)
            nose_area=img[nose_top_left[1]:nose_top_left[1]+nose_height,nose_top_left[0]:nose_top_left[0]+nose_width]
            _,mask=cv2.threshold(nose,25,255,cv2.THRESH_BINARY_INV)
            no_nose=cv2.bitwise_and(nose_area,nose_area,mask=mask)
            final_nose=cv2.add(no_nose,nose)
            img[nose_top_left[1]:nose_top_left[1] + nose_height, nose_top_left[0]:nose_top_left[0] + nose_width]=final_nose

        cv2.imshow('frame', img)
        #rcv2.imshow('nose',nose)
        key=cv2.waitKey(1)
        if key==27:
            break


cap.release()
cv2.destroyAllWindows()