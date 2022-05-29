# sudoku-solver

An app to detect and solve sudokus using your smartphone camera

## Installation

- install [python 3.10](https://www.python.org/downloads)
- install [pip](https://pypi.python.org/pypi/pip)
- install [pipenv](https://pipenv.pypa.io/en/latest/)
- run in project root:
  - `pipenv install --dev` to install dependencies
  - `pipenv shell` to activate virtualenv

## Development

- `./app` is the location of the app

to run the app:

- `pipenv shell` to ensure the virtualenv is activated
- `cd app` to enter the app directory
- `python main.py`

the app accepts one argument:

- `-d` or `--desktop` to run the app in desktop mode. If the flag is not present, the app will run in mobile mode.
- example: `python main.py -d`

to compile the app for android:

- connect your android phone to your computer
- turn on developer mode on your android phone
- enable USB debugging on your android phone through developer options
- run `buildozer android debug deploy` in `./app`
- wait for the app to install and then run `adb logcat -s "python"` to see the output (install adb if you don't have it)
- click on the app in the android phone

- alternatively, you can use `buildozer android debug deploy run` to immediately run the app, followed by `adb logcat -s "python"`

More information on [buildozer](https://github.com/kivy/buildozer)
