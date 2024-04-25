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


    #Runs the yolov8 model to detect object and return the resulting rectangles surrounding them
    def predict(self, image):
            #verbose refers to printing default outputs to the terminal
            #classes refers to what object the model is searching for, 0 is people
        results = self.model(image, verbose=False, classes=0)
        return results

    #Plots the boxes around the objects, incorparates the class that tracks the objects
    def plot_boxes(self, results, image):
        #define parameters for boxes, image input and line width
        self.annotator = Annotator(image, 3)
        self.annotatorThick = Annotator(image, 13)

        #Extract boxes from results
        boxes = results[0].boxes.xyxy.cpu()

        #For each box, add the box to the image that will be displayed to the user
        for box in boxes:
            self.annotator.box_label(box, color=(255, 0, 255))
        
        #Draw the simon boxes
        #Dimensions = xyxy
        self.annotatorThick.box_label([100, 300, 500, 475], color = (0, 0, 0))
        #Remember left and right are flipped, because camera is mirrored
        self.annotator.box_label([100, 300, 297, 385], color = (0, 255, 0)) #Green, top right 
        self.annotator.box_label([303, 300, 500, 386], color = (0, 0, 255)) #Red, top left
        self.annotator.box_label([100, 390, 298, 475], color = (0, 255, 255)) #Yellow, bottom right
        self.annotator.box_label([303, 390, 500, 475], color = (255, 0, 0)) #blue, bottom left

        #Flip image, so output mirrors movements
        image = cv2.flip(image, 1)

        return image
    
    #Displays a given text in the top right of the output
    def displayMessage(self, image, text):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.rectangle(image, (10, 60 - text_size[1]), (30 + text_size[0], 80), (255, 255, 255), -1)
        cv2.putText(image, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        return image

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
            image = self.plot_boxes(results, image)
            image = self.displayMessage(image, text = f'Press q to quit')

            #Display image
            cv2.imshow('Camera Test', image)
            
            #Allows user to quit program by pressing q
            if (cv2.waitKey(25) & 0xFF) == ord('q'):
                break
        
        #Once user exits program, release the capture and destroy the window    
        cap.release()
        cv2.destroyAllWindows()

""" Main Code """
if __name__ == "__main__":
    (H, W) = (None, None)
    detector = ObjectDetection(capture_index=0)
    detector()