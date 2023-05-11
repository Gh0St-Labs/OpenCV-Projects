import cv2 as cv
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, imageMode=False, maxHands=2, modelComplexity=1, minDetectionConfidence=0.5,
                 minTrackingConfidence=0.5):
        self.imageMode = imageMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.imageMode, self.maxHands, self.modelComplexity,
                                        self.minDetectionConfidence, self.minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frameObject, draw=True):
        frameRGB = cv.cvtColor(frameObject, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for handLandmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frameObject, handLandmark,
                                               self.mpHands.HAND_CONNECTIONS)
        return frameObject

    def findLandmark(self, frameObject, landmarkNo=0, draw=True):
        landmarkList = []
        if self.results.multi_hand_landmarks:
            userLandmark = self.results.multi_hand_landmarks[landmarkNo]
            for indexID, landMark in enumerate(userLandmark.landmark):
                height, width, channel = frameObject.shape
                cx, cy = int(landMark.x * width), int(landMark.y * height)
                landmarkList.append([indexID, cx, cy])
                if draw:
                    cv.circle(frameObject, (cx, cy), 15, (0, 255, 255), 2)
        return landmarkList

def main():
    previousTime = 0
    currentTime = 0
    video = cv.VideoCapture(0)

    detector = HandDetector()

    while True:
        bool, frame = video.read()

        frameObject = detector.findHands(frameObject=frame, draw=True)
        positionList = detector.findLandmark(frame)

        currentTime = time.time()  # Local Current Time
        FPS = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv.putText(frame, str(int(FPS)), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        cv.imshow('Video', frameObject)
        cv.waitKey(1)


if __name__ == '__main__':
    main()