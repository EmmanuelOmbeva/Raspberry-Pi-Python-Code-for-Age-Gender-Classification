import cv2
 
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes
 
faceProto = "/home/mypi/Age_Gender_Classfication/opencv_face_detector.pbtxt"
faceModel = "/home/mypi/Age_Gender_Classfication/opencv_face_detector_uint8.pb"
ageProto = "/home/mypi/Age_Gender_Classfication/age_deploy.prototxt"
ageModel = "/home/mypi/Age_Gender_Classfication/age_net.caffemodel"
genderProto = "/home/mypi/Age_Gender_Classfication/gender_deploy.prototxt"
genderModel = "/home/mypi/Age_Gender_Classfication/gender_net.caffemodel"
 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
 
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
 
video = cv2.VideoCapture(0)
padding = 20
 
hasFrame, frame = video.read()
if not hasFrame:
    print("Couldn't read a frame from the video.")
    exit()
 
desired_window_width = 480
ratio = desired_window_width / frame.shape[1]
height = int(frame.shape[0] * ratio)
 
while True:
    hasFrame, frame = video.read()
    if not hasFrame:
        break
    frame = cv2.resize(frame, (desired_window_width, height))
 
    resultImg, faceBoxes = highlightFace(faceNet, frame)
 
    for faceBox in faceBoxes:
        face = frame[
            max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
            max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)
        ]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
 
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
 
        text = "{}:{}".format(gender, age)
        cv2.putText(resultImg, text, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
 
cv2.destroyAllWindows()
video.release()
