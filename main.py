# We will be using the MediaPipe Module developed by Google for enhanced hand tracking bases.
# The Two Main Backends (infrastructures) involved in Hand Tracking
# 1. Palm Detection <> Finds the palm of the hand from any given source and crops the hand wisely
# 2. Hand Landmarks <> Positions 21 different landmarks spread across the cropped hand.
import cv2 as cv
import mediapipe as mp
import time

# <><><>IMPORTANT NOTICE<><><> After finishing this Project, we are going to be converting
# this into our own Hand Tracking Module which can get precise points out of the 21 points
# drawn in our hand. Refer the Inner for loop in the nested for loop defined in the While Loop.

video = cv.VideoCapture(0)
# This allows the OpenCV framework to get hold of your camera.
# There are two arguments in this method (VideoCapture), 0 and 1. 0 is for inbuilt webcams
# and 1 is for Connected Webcams. It is preferred to switch between 0 and 1 to find the appropriate
# one.

# <><><> STEP 1: Instantiate the Hands() Method from MediaPipe and the Drawing Utilities

mpHands = mp.solutions.hands
# Creating an instance to use it to function the 'Hands' method
hands = mpHands.Hands()
# The parameters in Hands()
# 1. static_image_mode
# This tells the webcam to either track and detect hands if the value given is False (set default)
# or to only detect hands if the value is set to True. Only detecting hands is a bit slower, so
# we can leave the value as True
# 2. max_num_hands
# This is also set to 2 by default. We can leave it for noe
# 3. min_detecting_confidence
# It has been set to 50% for now (0.5), so we can leave it as such
# 4. min_track_confidence
# Same as min_detecting_confidence
# As every value in the parameters have not been changed and set to default, we can just leave
# a raw method.
mpDraw = mp.solutions.drawing_utils
# This instance is for drawing all the 21 points / landmarks

# <><><> Optional STEP : Set the Previous and the Current Time for the FPS
# For displaying the FPS
# We need two instances, one for the previous time and one for the current time. SET THEM TO 0
previousTime = 0
currentTime = 0


# This 'while' loop is to capture the video FOREVER (use Common Sense)
while True:
    bool, frame = video.read()
    # 'bool' represents the two boolean values, true and false. When something is seen in the
    # video captured, the 'bool' is set to true, else it's false.
    # 'frame' is the Frame object. It holds every single frame made in the video captured.

    # We have to convert the Frame Object 'frame' to RGB before supplying it to the tools
    # of hand detection and the method 'Hands()'

    # <><><> STEP 2: Convert the Frame from BGR to RGB and process the image to get the landmarks
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    # The 'results' instance processes the RGB Frame Object which tracks and detects any hands
    # in the Frame Object
    # TO CHECK WHETHER AN HAND HAS BEEN DETECTED, WE CAN PRINT THE LANDMARKS USING THE 'results'
    # INSTANCE: Using the method, .multi_hand_landmarks
    # print(results.multi_hand_landmarks)

    # <><><> STEP 3: Iterate over each dictionary of X, Y and Z values which are wrapped around
    # a list IF there are any landmarks detected (landmarks are the points in the hands).
    if results.multi_hand_landmarks:
        # Iterating over each dictionary which contains the x, y and z landmarks, NOT ITERATING
        # OVER THE X, Y AND Z VALUES!
        for handLandmark in results.multi_hand_landmarks:

            # <><><> STEP 4: Draw the landmarks using the drawing utility instantiated in Step 1
            mpDraw.draw_landmarks(frame, handLandmark, mpHands.HAND_CONNECTIONS)
            # This is the method used to draw the landmarks >>> .draw_landmarks
            # It TAKES IN THE BGR Frame Object, NOT THE RGB Frame Object, as the RGB Frame Object
            # IS ONLY USED FOR DETECTING LANDMARKS (refer the 'results' instance)
            # First Argument: BGR Frame Object
            # Second Argument: The Dictionary which contains the X, Y and Z values (landmarks)
            # Third Argument: The method connects the points indicated on the Hand detected.
            # <><>RULE<><> MAKE SURE TO USE THE Instance which is going to be used to function the
            # Hands() METHOD ('mpHands', NOT 'Hands')
            # For getting precise points out of the 21 points/landmarks
            # To get the precise point, we need some startup information on the landmarks.
            # We can retrieve two values from the Landmark, one of which is the Index ID and the
            # other one is the Landmark value, which consists of the X, Y and Z values.

            # <><><> STEP 5: Iterate over each X, Y ans Z values in the dictionary with their
            # corresponding index values
            for indexID, landMark in enumerate(handLandmark.landmark):
                # The .landmark method gives only the landmark value
                # indexID represents the points on the hand (21 points) which are landmark-ed.
                # This is done due to the enumeration of the landmark values.
                # NOW WE NEED THE CX and the CY Values for the appropriate point derivation
                # To do that:

                # <><><> STEP 6: Set the width, height and the channels using the Frame Object's shape
                # for initiating the CX and the CY values
                height, width, channel = frame.shape
                cx = int(landMark.x * width)
                cy = int(landMark.y * height)
                # This is for the outer X and the Y values. Refer the Shapes and Text Folder
                # in the Acer Laptop under the OpenCV_Learn Folder.
                # This type of CX, and CY values are given as a value for the Point 2 while
                # drawing a rectangle.
                # landMark.x returns the landmark's X value. Applicable to Y and X values, too!

                # <><><> Optional Step: Draw a circle around the specific landmark number
                if indexID == 10:
                    cv.circle(frame, (cx, cy), 15, (0,255,255), 2)

    # <><><> Optional Step: Set the FPS using the Previous and the Current Time
    # THIS IS THE SYNTAX FOR CREATING THE FPS. JUST FOLLOW / COPY THIS.
    currentTime = time.time() # Our Local Current Time
    FPS = 1/(currentTime-previousTime)
    previousTime = currentTime

    # <><><> Optional Step: Show the FPS on the Webcam
    cv.putText(frame, str(int(FPS)), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    # <><><> STEP 7: Show the Frame on the window.
    cv.imshow('Video', frame)
    cv.waitKey(1)


