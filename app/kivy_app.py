from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder


import cv2 as cv
import tensorflow as tf
import os 
import numpy as np
from our_classifier.digit_classifier import DigitClassifier
from utils.utils import read_img
from utils.extract_grid import extract_grid
from utils.grid_to_array import grid_to_array


screens = Builder.load_file("screens.kv")
# 9x9 matrix to save the predicted digits
predicted_digits = None
# 9x9 matrix to save the solution of the sudoku
solution = None

class MainPage(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        # define our video capture object 
        self.capture = cv.VideoCapture(0)

    # event fired when screen is displayed
    def on_enter(self, *args):
        self.schedule = Clock.schedule_interval(self.update_camera, 1.0 / 33.0)

    # event fired when leaving the screen
    def on_leave(self, *args):
        if (self.schedule):
            self.schedule.cancel()
            self.schedule = None

    # function called to update the camera image constantly 
    def update_camera(self, *args):
        # read frame from our video capture object
        ret, frame = self.capture.read()

        # flip the image horizontally 
        buffer = cv.flip(frame, 0).tostring()
        # create a texture with size of the frame
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        # convert the image buffer to a texture
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        # setting the texture for our sudoku image (define in screens.kv)
        self.ids.sudoku.texture = texture  

    # function called when "use this photo" button is pressed
    def predict_numbers(self):
        # save image from camera to a path
        ret, frame = self.capture.read()
        cv.imwrite("./images/input_image.jpg", frame)

        # get image from the path 
        img = read_img("./images/input_image.jpg")
        # extract grid from the image
        img_grid = extract_grid(img)
        # get the predicted array of digits
        predicted_digits = grid_to_array(img_grid)

        # change screen to adjustment page
        self.manager.current = 'adjustmentPage'
        self.manager.transition.direction = 'left'


class AdjustmentPage(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.textInputs =  None
        

    def addInputBoxes(self):
        for i in range(10): 
            for j in range(10):
                self.textInputs[i][j] = TextInput(text=predicted_digits[i][j], multiline=False)
                self.add_widget(self.textInputs[i][j])
    
    def get_adjustment(self):

        for i in range(10): 
            for j in range(10):
                predicted_digits[i][j] = self.textInputs[i][j].text 

        # TODO: solve the sudoku (using sudoku class) and save it to solution

        # change screen to adjustment page
        self.manager.current = 'solutionPage'
        self.manager.transition.direction = 'left'

class SolutionPage(Screen):
    def on_enter(self, *args):
        # TODO: show the solution (as image (id in screens.kv / labels )

        pass

        
class SudokuSolverApp(App):
    def build(self):
        return screens

if __name__ == '__main__':
    SudokuSolverApp.run()
