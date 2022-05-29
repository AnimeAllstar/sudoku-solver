import os

# allow arguments to be passed to the app
os.environ["KIVY_NO_ARGS"] = "1"

import cv2 as cv
from pathlib import Path
import numpy as np
from solver.sudoku import Sudoku
from utils.utils import read_img
from utils.extract_grid import extract_grid
from utils.grid_to_array import grid_to_array
from functools import partial
import argparse

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button

from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.core.window import Window


# define widgets properties
screens = Builder.load_file("screens.kv")
# set window background color
Window.clearcolor = (1, 1, 1, 1)
# 9x9 matrix to save the predicted digits
predicted_digits = np.zeros(shape=(9, 9))
# 9x9 matrix to save the solution of the sudoku
solution = np.zeros(shape=(9, 9))


class Cell(ToggleButton):
    pass


class ButtonInput(Button):
    pass


class ButtonCell(Cell):
    pass


class LabelCell(Cell):
    pass


class NoSolutionPage(Screen):
    pass


class NoSudokuPage(Screen):
    pass


class CameraPage(Screen):
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

    # update the camera image (called constantly in CameraPage)
    def update_camera(self, *args):

        if not self.capture.isOpened():
            print("Error opening camera")
            self.on_leave()
            return

        # read frame from our video capture object
        ret, frame = self.capture.read()
        # flip the image horizontally
        buffer = cv.flip(frame, 0).tobytes()
        # create a texture with size of the frame
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        # convert the image buffer to a texture
        texture.blit_buffer(buffer, colorfmt="bgr", bufferfmt="ubyte")
        # setting the texture for our sudoku image (define in screens.kv)
        self.ids.sudoku.texture = texture

    # capture image from camera and predict numbers (use this photo button)
    def predict_numbers(self, test=False):
        global predicted_digits

        if not test:
            if not self.capture.isOpened():
                print("Error opening camera")
                return

            # cv.imwrite("./temp/images/input_image.jpg", frame)
            ret, frame = self.capture.read()
            img = frame
        else:
            # ensure directory exists
            Path("./temp/images").mkdir(parents=True, exist_ok=True)

            try:
                img = read_img("./temp/images/test_image.jpg")
            except:
                print("Error reading image")
                return

        # extract grid from the image
        img_grid = extract_grid(img)

        if img_grid is not None:
            # get the predicted array of digits
            predicted_digits = grid_to_array(img_grid, isMobile=isMobile)

            # change screen to CorrectionPage
            self.manager.current = "correctionPage"
        else:
            self.manager.current = "noSudokuPage"


class CorrectionPage(Screen):
    def __init__(self, **kw):
        self.cells = np.ndarray(shape=(9, 9), dtype=ButtonCell)
        self.selected = None
        self.location = None
        self._keyboard = None
        super().__init__(**kw)

    # bind keyboard events when screen is displayed
    def on_pre_enter(self, *args):
        if isMobile:
            pass
        else:
            self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
            self._keyboard.bind(on_key_down=self._on_keyboard_down)
        return super().on_pre_enter(*args)

    # release keyboard when leaving the screen
    def on_leave(self, *args):
        if self._keyboard:
            self._keyboard.release()
            return super().on_leave(*args)

    # callback function that will be called when the keyboard is closed
    def _keyboard_closed(self):
        if self._keyboard:
            self._keyboard.unbind(on_key_down=self._on_keyboard_down)
            self._keyboard = None

    # called when a key is pressed
    # if the key is a valid number, update the selected cell
    # if the key is a direction, simulate a click in the direction
    # if the key is backspace or delete, clear the selected cell
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if self.selected:
            row, col = self.location
            code, value = keycode
            if value == "right":
                self.handle_cell_press(
                    self.cells[row, (col + 1) % 9], row, (col + 1) % 9
                )
            elif value == "left":
                self.handle_cell_press(
                    self.cells[row, (col - 1) % 9], row, (col - 1) % 9
                )
            elif value == "up":
                self.handle_cell_press(
                    self.cells[(row - 1) % 9, col], (row - 1) % 9, col
                )
            elif value == "down":
                self.handle_cell_press(
                    self.cells[(row + 1) % 9, col], (row + 1) % 9, col
                )
            elif ord("1") <= code <= ord("9"):
                self.selected.text = value
            elif value == "backspace" or value == "delete":
                self.selected.text = ""
        else:
            # if no cell is selected, do not accept any key, it will be used by the system.
            return False

    # update the selected cells value with the input clicked
    def update_cell(self, input):
        if self.selected:
            self.selected.text = input.text

    # clear the selected cell
    def clear_cell(self, input):
        if self.selected:
            self.selected.text = ""

    # handle cell press event for cell from the grid
    def handle_cell_press(self, cell, row, col):
        # if a cell is already selected, unselect it
        if self.selected:
            self.selected.state = "normal"
            self.selected.color = (0, 0, 0, 1)

        # set self.selected to cell
        self.selected = cell
        self.selected.state = "down"
        self.selected.color = (1, 1, 1, 1)
        self.location = (row, col)

    # add input buttons
    def add_input_buttons(self):
        for i in range(9):
            self.inputs.add_widget(
                ButtonInput(text=str(i + 1), on_press=self.update_cell)
            )
        self.inputs.add_widget(ButtonInput(text="X", on_press=self.clear_cell))

    # add buttons representing each cell for the user to click on to select
    def add_cell_buttons(self):
        for i in range(9):
            for j in range(9):
                txt = "" if predicted_digits[i][j] == 0 else str(predicted_digits[i][j])
                self.cells[i][j] = ButtonCell(text=txt)
                callback = partial(self.handle_cell_press, row=i, col=j)
                self.cells[i][j].bind(on_press=callback)
                self.grid.add_widget(self.cells[i][j])

    # update the input boxes whenever the screen is displayed
    def on_enter(self, *args):
        for i in range(9):
            for j in range(9):
                txt = "" if predicted_digits[i][j] == 0 else str(predicted_digits[i][j])
                self.cells[i][j].text = txt

    # solve the sudoku
    def solve(self):
        global predicted_digits, solution

        for i in range(9):
            for j in range(9):
                if self.cells[i][j].text == "":
                    predicted_digits[i][j] = 0
                else:
                    predicted_digits[i][j] = self.cells[i][j].text

        sudoku = Sudoku(predicted_digits)
        solvable = sudoku.solve()

        # reset selected cell to normal
        if self.selected is not None:
            self.selected.state = "normal"
            self.selected.color = (0, 0, 0, 1)

        if solvable:
            solution = sudoku.solution
            # change screen to solutionPage
            self.manager.current = "solutionPage"
        else:
            self.manager.current = "noSolutionPage"


class SolutionPage(Screen):
    def __init__(self, **kw):
        self.solution_labels = np.ndarray(shape=(9, 9), dtype=Label)
        super().__init__(**kw)

    # add labels to show the solution of the sudoku
    def showSolution(self):
        for i in range(9):
            for j in range(9):
                self.solution_labels[i][j] = LabelCell(text="")
                self.grid.add_widget(self.solution_labels[i][j])

    # update the labels whenever the screen is displayed
    def on_enter(self, *args):
        for i in range(9):
            for j in range(9):
                self.solution_labels[i][j].text = str(solution[i][j])


class SudokuSolverApp(App):
    def build(self):
        global isMobile

        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--desktop", action="store_false")
        args = parser.parse_args()
        isMobile = args.desktop

        sm = ScreenManager(transition=NoTransition())
        sm.add_widget(CameraPage())
        sm.add_widget(CorrectionPage())
        sm.add_widget(SolutionPage())
        sm.add_widget(NoSolutionPage())
        sm.add_widget(NoSudokuPage())
        return sm


if __name__ == "__main__":
    SudokuSolverApp().run()
