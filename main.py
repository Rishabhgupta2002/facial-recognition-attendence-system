import face_recognition
import cv2
import numpy
import csv
from datetime import datetime

videocap = cv2.VideoCapture(0)
now=datetime.now()

  # data of the known students
rishabh=face_recognition.load_image_file("faces/rishabh.jpg")
shiv=face_recognition.load_image_file("faces/shiv.jpg")
vivek=face_recognition.load_image_file("faces/vivek.jpg")
sajal=face_recognition.load_image_file("faces/sajal.jpg")
manish=face_recognition.load_image_file("faces/manish.jpg")

rishabh_encode=face_recognition.face_encodings(rishabh)[0]
shiv_encode=face_recognition.face_encodings(shiv)[0]
vivek_encode=face_recognition.face_encodings(vivek)[0]
sajal_encode=face_recognition.face_encodings(sajal)[0]
manish_encode=face_recognition.face_encodings(manish)[0]

known_faces_encodings=[rishabh_encode,shiv_encode,vivek_encode,sajal_encode,manish_encode]
known_faces_names=["Rishabh Gupta","Shiv Trivedi","Vivek Upadhyay","Sajal Sahu","Manish Sharma"]
students=known_faces_names.copy()

 # processing the details of approaching students
face_locations=[]
face_encodings=[]


date=now.strftime("%d-%m-%Y")
f = open(f"{date}.csv","w+",newline="")
lnwriter=csv.writer(f)

while 1:
    _, frame=videocap.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.2,fy=0.2)
    rgb_smallframe=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    face_locations=face_recognition.face_locations(rgb_smallframe)
    face_encodings=face_recognition.face_encodings(rgb_smallframe,face_locations)

    for face_encoding in face_encodings:
        match=face_recognition.compare_faces(known_faces_encodings,face_encoding)
        face_dist=face_recognition.face_distance(known_faces_encodings,face_encoding)
        best_match_index=numpy.argmin(face_dist)

        if(match[best_match_index]):
            name=known_faces_names[best_match_index]


        #display name of the person
        if name in known_faces_names:
            font=cv2.FONT_HERSHEY_SIMPLEX
            bottomleft=(10,50)
            fontscale=1.0
            font_color=(0,0,0)
            thickness=2
            linetype=3
            cv2.putText(frame,name,bottomleft,font,fontscale,font_color,thickness,linetype)

        # remove student after taking attendance
        if name in students:
            students.remove(name)
            current_time=now.strftime("%H:%M:%S")
            lnwriter.writerow([name,current_time])


    cv2.imshow("Face Attendence",frame)
    if(cv2.waitKey(1) & 0xFF==ord("q")):
        break

videocap.release()
cv2.destroyAllWindows()
f.close()