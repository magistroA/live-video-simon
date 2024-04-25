import torch
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import random
import sounddevice as sd

#Detects people as objects from live video footage, will automatically open the webcam of a laptop
class ObjectDetection:
    def __init__(self, capture_index):
        #Default Parameters
        self.capture_index = capture_index

        #Model Information
        self.model = YOLO("yolov8n.pt")
        #Initialize Annotator for drawing boxes
        self.annotator = None
        #Device Info
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #Game Setup Variables
        self.Sequence = []
        self.patternLen = 3
        self.delay = 18
        self.currentIndex = 0
        self.userInput = [0,0,0,0]
        self.currentInput = 0
        self.score = 0

        #Holds the number of frames that have been processed, used to hold in phases for set periods of time
        self.counter = 0
        self.inputDelayCounter = 0

        #Phases of the game
        #0  - Welcome message
        #1  - Round Start
        #2  - Increment Sequence Value
        #3  - Play Tone
        #4  - End of Sequence
        #5  - Read from User
        #6  - User made an input
        #7  - End of user selected box
        #8  - Correct Input
        #9  - Set next round inputs
        #10 - User made an error or timed out
        #11 - Game Score
        #12 - End of Game Prompt
        self.currentPhase = 0

    #Runs the yolov8 model to detect object and return the resulting rectangles surrounding them
    def predict(self, image):
            #verbose refers to printing default outputs to the terminal
            #classes refers to what object the model is searching for, 0 is people
        results = self.model(image, verbose=False, classes=0)
        return results

    #Plots the boxes around the objects, incorparates the class that tracks the objects
    def plot_boxes(self, results, image, num, checkPlayer):
        #define parameters for boxes, image input and line width
        self.annotator = Annotator(image, 3)
        self.annotatorThick = Annotator(image, 13)

        #Extract boxes from results
        boxes = results[0].boxes.xyxy.cpu()

        #For each box, add the box to the image that will be displayed to the user
        for box in boxes:
            self.annotator.box_label(box, color=(255, 0, 255))

            if checkPlayer == True:
                self.checkIfInBox(box)
        
        #Draw the simon boxes
        #Dimensions = xyxy
        self.annotatorThick.box_label([100, 300, 500, 475], color = (0, 0, 0))
        #Remember left and right are flipped, because camera is mirrored
        self.annotator.box_label([100, 300, 297, 385], color = (0, 255, 0)) #Green, top right 
        self.annotator.box_label([303, 300, 500, 386], color = (0, 0, 255)) #Red, top left
        self.annotator.box_label([100, 390, 298, 475], color = (0, 255, 255)) #Yellow, bottom right
        self.annotator.box_label([303, 390, 500, 475], color = (255, 0, 0)) #blue, bottom left

        #Will indicate the correct square in the sequence, as passed into the function
        if(num != -1):
            self.drawSquare(image, num)

        #Flip image, so output mirrors movements
        image = cv2.flip(image, 1)

        return image
    
    #Count how many frames the player stays in a box
    def checkIfInBox(self, box):
        x = (box[0] + box[2]) / 2
        y = box[3]

        #top left
        if (125 < x < 275) and (300 < y < 400):
            self.userInput = [(self.userInput[i] + 1)  if i == 1 else 0 for i in range(4)]

        #top right
        elif (325 < x < 475) and (300 < y < 400):
            self.userInput = [(self.userInput[i] + 1)  if i == 0 else 0 for i in range(4)]

        #bottom left
        elif (125 < x < 275) and (385 < y < 500):
            self.userInput = [(self.userInput[i] + 1)  if i == 3 else 0 for i in range(4)]

        #bottom right
        elif (325 < x < 475) and (385 < y < 500):
            self.userInput = [(self.userInput[i] + 1)  if i == 2 else 0 for i in range(4)]
        
        #resets counter if user isn't in a box
        else:
            self.userInput = [0 for i in range(4)]
    
    #Displays a given text in the top right of the output
    def displayMessage(self, image, text):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.rectangle(image, (10, 60 - text_size[1]), (30 + text_size[0], 80), (255, 255, 255), -1)
        cv2.putText(image, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        return image
    
    #Generates a new random sequence of values
    def generateSequence(self):
        options = [1,2,3,4]
        self.Sequence = []

        for i in range(self.patternLen):
            self.Sequence.append(random.choice(options))
    
    #Draws a thinker line around the current color in the sequence
    def drawSquare(self, image, num):
        self.annotatorThick = Annotator(image, 13)

        match num:
            case 1:
                self.annotatorThick.box_label([303, 300, 500, 386], color = (0, 0, 255)) #Red, top left

            case 2:
                self.annotatorThick.box_label([100, 300, 297, 385], color = (0, 255, 0)) #Green, top right

            case 3:
                self.annotatorThick.box_label([303, 390, 500, 475], color = (255, 0, 0)) #blue, bottom left

            case 4:
                self.annotatorThick.box_label([100, 390, 298, 475], color = (0, 255, 255)) #Yellow, bottom right

        return image
    
    #Playes a tone corresponding to the current color in the sequence
    def playTone(self, num):
        match num:
            case 1:
                tone = np.sin(2 * np.pi * 800 * np.arange(0, 1, 1/44100))

            case 2:
                tone = np.sin(2 * np.pi * 600 * np.arange(0, 1, 1/44100))

            case 3:
                tone = np.sin(2 * np.pi * 400 * np.arange(0, 1, 1/44100))

            case 4:
                tone = np.sin(2 * np.pi * 200 * np.arange(0, 1, 1/44100))

        sd.play(tone, 44100)


    #Function that runs constantly when the object is called
    def __call__(self):
        #Open the webcam for video capture
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        
        #Run until the user hits q to exit the program
        #Around 10 fps
        while True:
            #Read from Web Cam
            ret, image = cap.read()
            assert ret
            #Run prediction model to find people
            results = self.predict(image)
            
            match self.currentPhase:
                #Start of Game
                case 0:
                    image = self.plot_boxes(results, image, -1, False)
                    image = self.displayMessage(image, text = f'Welcome to Simon')
                    
                    if self.counter < 20:
                        self.counter += 1
                    else:
                        self.counter = 0
                        self.currentPhase = 1

                #Round Start
                case 1:
                    image = self.plot_boxes(results, image, -1, False)

                    self.generateSequence()
                    image = self.displayMessage(image, text = f'Wait for Sequence to Complete')

                    self.currentPhase = 2

                #Increment Sequence Value
                case 2:
                    image = self.plot_boxes(results, image, self.Sequence[self.currentIndex], False)
                    image = self.displayMessage(image, text = f'Wait for Sequence to Complete')

                    if self.counter == 1:
                        self.counter += 2
                        self.currentPhase = 3
                    elif self.counter < self.delay:
                        self.counter += 1
                    else:
                        self.counter = 0
                        self.currentPhase = 4

                #Play Tone
                case 3:
                    image = self.plot_boxes(results, image, self.Sequence[self.currentIndex], False)
                    image = self.displayMessage(image, text = f'Wait for Sequence to Complete')
                    self.playTone(self.Sequence[self.currentIndex])

                    self.currentPhase = 2

                #End of Sequence
                case 4:
                    image = self.plot_boxes(results, image, self.Sequence[self.currentIndex], False)
                    image = self.displayMessage(image, text = f'Wait for Sequence to Complete')

                    self.counter = 0

                    #Continue with current Sequence
                    if self.currentIndex < self.patternLen - 1:
                        self.currentIndex += 1
                        self.currentPhase = 2

                    #Read User Input
                    else:
                        self.currentIndex = 0
                        self.currentPhase = 5

                #Read User Input
                case 5:
                    image = self.plot_boxes(results, image, -1, True)
                    image = self.displayMessage(image, text = f'Repeat the Sequence')

                    for i, numFrames in enumerate(self.userInput):
                        #User has selected that box
                        if numFrames > 7:
                            self.currentPhase = 6
                            self.currentInput = i + 1

                    #timeout
                    if self.counter < (self.delay * self.patternLen):
                        self.counter += 1
                    else:
                        self.counter = 0
                        self.currentPhase = 10

                #User selected a box
                case 6:
                    image = self.plot_boxes(results, image, self.currentInput, False)
                    image = self.displayMessage(image, text = f'Repeat the Sequence')

                    if(self.inputDelayCounter == 1):
                        self.playTone(self.currentInput)

                    if self.inputDelayCounter < 10:
                        self.inputDelayCounter += 1
                    else:
                        self.inputDelayCounter = 0
                        self.currentPhase = 7
                
                #End of user selected box
                case 7:
                    image = self.plot_boxes(results, image, self.currentInput, False)
                    image = self.displayMessage(image, text = f'Repeat the Sequence')

                    self.userInput = [0 for i in range(4)]

                    if self.currentInput == self.Sequence[self.currentIndex]:
                        self.currentIndex += 1
                        self.score += 1

                        if self.currentIndex == len(self.Sequence):
                            self.currentPhase = 8
                        else:
                            self.currentPhase = 5

                    else:
                        self.counter = 0
                        self.currentPhase = 10

                #Correct Input
                case 8:
                    image = self.plot_boxes(results, image, -1, False)
                    image = self.displayMessage(image, text = f'Correct!')

                    if self.counter < 5:
                        self.counter += 1
                    else:
                        self.counter = 0
                        self.currentPhase = 9
                
                #Set next round inputs
                case 9:
                    image = self.plot_boxes(results, image, -1, False)
                    image = self.displayMessage(image, text = f'Correct!')

                    self.patternLen += 1

                    if self.delay >= 4:
                        self.delay -= 2

                    self.counter = 0
                    self.currentIndex = 0

                    self.currentPhase = 1

                #User made an error or timed out
                case 10:
                    image = self.plot_boxes(results, image, -1, False)
                    image = self.displayMessage(image, text = f'Wrong')

                    if self.counter < 10:
                        self.counter += 1
                    else:
                        self.counter = 0
                        self.currentPhase = 11

                #Game Score
                case 11:
                    image = self.plot_boxes(results, image, -1, False)
                    image = self.displayMessage(image, text = ('Game Score : ' + str(self.score)))

                    if self.counter < 10:
                        self.counter += 1
                    else:
                        self.counter = 0
                        self.currentPhase = 12

                #End Game Prompt
                case 12:
                    image = self.plot_boxes(results, image, -1, False)
                    image = self.displayMessage(image, text = ('Press q to quit. Press r to replay'))


            #Display image
            cv2.imshow('Simon Game', image)
            
            #Allows user to quit program by pressing q
            if (cv2.waitKey(25) & 0xFF) == ord('q'):
                break
            if (cv2.waitKey(25) & 0xFF) == ord('r'):
                self.currentPhase = 0
        
        #Once user exits program, release the capture and destroy the window    
        cap.release()
        cv2.destroyAllWindows()

""" Main Code """
if __name__ == "__main__":
    (H, W) = (None, None)
    detector = ObjectDetection(capture_index=0)
    detector()