import cv2

cap = cv2.VideoCapture(0)
hip = cv2.VideoCapture(2)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitkey(1) * 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
