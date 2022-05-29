import numpy as np
from jnius import autoclass

# since tflite on android can only be performed using C++ / Java
# we need to acess java classes and use them during implementation

File = autoclass("java.io.File")
Interpreter = autoclass("org.tensorflow.lite.Interpreter")
InterpreterOptions = autoclass("org.tensorflow.lite.Interpreter$Options")
TensorBuffer = autoclass("org.tensorflow.lite.support.tensorbuffer.TensorBuffer")
ByteBuffer = autoclass("java.nio.ByteBuffer")


class DigitClassifierAndroid:
    def __init__(self, num_threads=None):
        model = File("./classifier/saved_model/digit_classifier.tflite")
        options = InterpreterOptions()
        if num_threads is not None:
            options.setNumThreads(num_threads)
        self.interpreter = Interpreter(model, options)
        self.interpreter.allocateTensors()

    def get_output_buffer(self):
        y_shape = self.interpreter.getOutputTensor(0).shape()
        y_type = self.interpreter.getOutputTensor(0).dataType()
        self.y_buffer = TensorBuffer.createFixedSize(y_shape, y_type)
        self.y_buffer = self.y_buffer.getBuffer().rewind()

    def predict(self, X):
        X_buffer = ByteBuffer.wrap(X.tobytes())
        self.get_output_buffer()
        self.interpreter.run(X_buffer, self.y_buffer)
        prediction = np.array(self.y_buffer.getFloatArray())
        prediction = np.reshape(prediction)
        return np.argmax(prediction)
