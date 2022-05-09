from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.lang import Builder


import cv2 as cv
import tensorflow as tf
import os 
import numpy as np

screens = Builder.load_file("screens.kv")

class MainPage(Screen):
    pass

class AdjustmentPage(Screen):
    pass

class SolutionPage(Screen):
    pass

class SudokuSolverApp(App):
    def build(self):
        return screens

if __name__ == '__main__':
    SudokuSolverApp.run()
