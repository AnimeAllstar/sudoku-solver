import numpy as np


class Sudoku:
    def __init__(self, grid=None):
        self.solution = None
        self.N = 9

        if grid is not None:
            if isinstance(grid, np.ndarray):
                self.grid = grid
            else:
                self.grid = np.array(grid)

            if self.grid.shape != (self.N, self.N):
                raise ValueError(f"grid must be {self.N}x{self.N}")

            self.unsolved = self.grid.copy()
        else:
            self.grid = np.zeros((self.N, self.N), dtype=int)

    # setter
    def set(self, row, col, num):
        self.grid[row][col] = num

    # getter
    def get(self, row, col):
        return self.grid[row][col]

    # solve the sudoku using check_solvable function
    def solve(self):
        if self.solution is None:
            self.unsolved = self.grid.copy()
        if self.check_solvable(0, 0):
            self.solution = self.grid.copy()
            return True
        return False

    def check_solvable(self, row, col):
        if row == self.N - 1 and col == self.N:
            return True
        if col == self.N:
            row += 1
            col = 0
        if self.get(row, col) > 0:
            return self.check_solvable(row, col + 1)
        for num in range(1, self.N + 1, 1):
            if self.check_solvable_helper(row, col, num):
                self.set(row, col, num)
                if self.check_solvable(row, col + 1):
                    return True
            self.set(row, col, 0)
        return False

    def check_solvable_helper(self, row, col, num):
        for x in range(9):
            if self.get(row, x) == num:
                return False

        for x in range(9):
            if self.get(x, col) == num:
                return False

        startRow = row - row % 3
        startCol = col - col % 3
        for i in range(3):
            for j in range(3):
                if self.get(i + startRow, j + startCol) == num:
                    return False
        return True

    # return solution
    def solution(self):
        return self.solution

    # print solution on console
    def show(self, solution=False):
        if solution:
            print(self.solution)
        else:
            print(self.unsolved)
