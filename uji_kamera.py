import cv2

CAMERA_SOURCE = 1  # coba 0,1,2,3
cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)

print("isOpened:", cap.isOpened())

while True:
    ok, frame = cap.read()
    print("read:", ok)
    if not ok:
        continue
    cv2.imshow("test-cam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
