from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.uix.textinput import TextInput
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder


import cv2 as cv
import tensorflow as tf
import os
from pathlib import Path
import numpy as np
from our_classifier.digit_classifier import DigitClassifier
from solver.sudoku import Sudoku
from utils.utils import read_img
from utils.extract_grid import extract_grid
from utils.grid_to_array import grid_to_array


screens = Builder.load_file("screens.kv")
# 9x9 matrix to save the predicted digits and the solution of the sudoku
global predicted_digits, solution
predicted_digits, solution = None, None


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
        if self.schedule:
            self.schedule.cancel()
            self.schedule = None

    # function called to update the camera image constantly
    def update_camera(self, *args):
        # read frame from our video capture object
        ret, frame = self.capture.read()

        # flip the image horizontally
        buffer = cv.flip(frame, 0).tobytes()
        # create a texture with size of the frame
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        # convert the image buffer to a texture
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        # setting the texture for our sudoku image (define in screens.kv)
        self.ids.sudoku.texture = texture

    # function called when "use this photo" button is pressed
    def predict_numbers(self, test=False):
        global predicted_digits

        # save image from camera to a path
        ret, frame = self.capture.read()

        # ensure directory exists
        Path("./temp/images").mkdir(parents=True, exist_ok=True)

        if not test:
            cv.imwrite("./temp/images/input_image.jpg", frame)
            # get image from the path
            img = read_img("./temp/images/input_image.jpg")
        else:
            img = read_img("./temp/images/test_image.png")

        # extract grid from the image
        img_grid = extract_grid(img)
        # get the predicted array of digits
        predicted_digits = grid_to_array(img_grid)

        # change screen to adjustment page
        self.manager.add_widget(AdjustmentPage())
        self.manager.current = 'adjustmentPage'


class AdjustmentPage(Screen):
    def __init__(self, **kw):
        self.text_inputs = np.ndarray(shape=(9, 9), dtype=TextInput)
        super().__init__(**kw)

    def addInputBoxes(self):
        global predicted_digits

        for i in range(9):
            for j in range(9):
                self.text_inputs[i][j] = TextInput(
                    text=str(predicted_digits[i][j]),
                    multiline=False,
                )
                self.grid.add_widget(self.text_inputs[i][j])

    def get_adjustment(self):
        global predicted_digits, solution
        for i in range(9):
            for j in range(9):
                predicted_digits[i][j] = self.text_inputs[i][j].text

        sudoku = Sudoku(predicted_digits)
        sudoku.solve()
        sudoku.show(solution=True)
        solution = sudoku.solution

        # change screen to adjustment page
        self.manager.add_widget(SolutionPage())
        self.manager.current = 'solutionPage'


class SolutionPage(Screen):
    def on_enter(self, *args):
        # TODO: show the solution (as image (id in screens.kv / labels)
        pass


class SudokuSolverApp(App):
    def build(self):
        sm = ScreenManager(transition=NoTransition())
        sm.add_widget(MainPage())
        return sm


if __name__ == '__main__':
    SudokuSolverApp().run()
