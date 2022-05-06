from sudoku import Sudoku as py_Sudoku
from solver.sudoku import Sudoku


def test():
    puzzle = py_Sudoku(3).difficulty(0.5)
    print("empty grid:")
    puzzle.show()
    solution = puzzle.solve()
    print("py_Sudoku solution:")
    solution.show()
    grid = None_to_zero(puzzle.board.copy())
    sudoku = Sudoku(grid)
    sudoku.solve()
    print("our solution:")
    sudoku.show()


def None_to_zero(grid):
    for i in range(9):
        for j in range(9):
            if grid[i][j] == None:
                grid[i][j] = 0
    return grid
