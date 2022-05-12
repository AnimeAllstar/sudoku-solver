from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.core.window import Window


import cv2 as cv
import os
from pathlib import Path
import numpy as np
from solver.sudoku import Sudoku
from utils.utils import read_img
from utils.extract_grid import extract_grid
from utils.grid_to_array import grid_to_array

# define widgets properties
screens = Builder.load_file("screens.kv")
# set window background color
Window.clearcolor = (1, 1, 1, 1)
# 9x9 matrix to save the predicted digits
predicted_digits = np.zeros(shape=(9, 9))
# 9x9 matrix to save the solution of the sudoku
solution = np.zeros(shape=(9, 9))

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

    # update the camera image (called constantly in main page)
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

    # capture image from camera and predict numbers (use this photo button)
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

        if img_grid is not None:
            # get the predicted array of digits
            predicted_digits = grid_to_array(img_grid)

            # change screen to adjustment page
            self.manager.current = 'adjustmentPage'
        else:
            self.manager.current = 'noSudokuPage'


class AdjustmentPage(Screen):
    def __init__(self, **kw):
        self.text_inputs = np.ndarray(shape=(9, 9), dtype=TextInput)
        super().__init__(**kw)

    # add input boxes for user to adjust the predicted numbers
    def addInputBoxes(self):
        for i in range(9):
            for j in range(9):
                txt = '' if predicted_digits[i][j] == 0 else str(predicted_digits[i][j])
                self.text_inputs[i][j] = TextInput(
                    text=txt,
                    multiline=False,
                    write_tab=False, 
                    halign='center'
                )
                self.grid.add_widget(self.text_inputs[i][j])

    # update the input boxes whenever the screen is displayed     
    def on_enter(self, *args):
        for i in range(9):
            for j in range(9):
                txt = '' if predicted_digits[i][j] == 0 else str(predicted_digits[i][j])
                self.text_inputs[i][j].text = txt

    # get adjustment from the user (confirm button)
    def get_adjustment(self):
        global predicted_digits, solution
        for i in range(9):
            for j in range(9):
                if self.text_inputs[i][j].text == '':
                    predicted_digits[i][j] = 0
                else:
                    predicted_digits[i][j] = self.text_inputs[i][j].text.strip()

        sudoku = Sudoku(predicted_digits)
        solvable = sudoku.solve()
        if solvable:    
            solution = sudoku.solution
            # change screen to adjustment page
            self.manager.current = 'solutionPage'
        else :
            self.manager.current = 'noSolutionPage'
        
        
class NoSolutionPage(Screen):
    pass

class NoSudokuPage(Screen):
    pass

class SolutionPage(Screen):
    def __init__(self, **kw):
        self.solution_labels = np.ndarray(shape=(9, 9), dtype=Label)
        super().__init__(**kw)

    # add labels to show the solution of the sudoku
    def showSolution(self):
        for i in range(9):
            for j in range(9):
                self.solution_labels[i][j] = Label(
                    text=' ', 
                    markup=True
                    )
                self.grid.add_widget(self.solution_labels[i][j])

    # update the labels whenever the screen is displayed 
    def on_enter(self, *args):
        for i in range(9):
            for j in range(9):
                if predicted_digits[i][j] != 0:
                    self.solution_labels[i][j].text = '[color=434445]' + str(solution[i][j])+ '[/color]'
                else:
                    self.solution_labels[i][j].text = '[color=e05f38]' + str(solution[i][j])+ '[/color]'
                


class SudokuSolverApp(App):
    def build(self):
        sm = ScreenManager(transition=NoTransition())
        sm.add_widget(MainPage())
        sm.add_widget(AdjustmentPage())
        sm.add_widget(SolutionPage())
        sm.add_widget(NoSolutionPage())
        sm.add_widget(NoSudokuPage())       
        return sm

if __name__ == '__main__':
    SudokuSolverApp().run()
