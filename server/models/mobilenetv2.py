import keras

IMG_SIZE = (160, 160)
IMG_SHAPE = (160, 160, 3)
MEAN = [0.23740229, 0.23729787, 0.23700129]
STD = [0.23173477, 0.23151317, 0.23122775]

preprocess_input = keras.applications.mobilenet_v2.preprocess_input

inputs = keras.Input(shape=IMG_SIZE)
base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False
)

base_model.trainable = False
global_average_layer = keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)

class ImagePreprocessLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(ImagePreprocessLayer, self).__init__()
    def call(self, img):
      print(type(img), img)
      img = img.numpy()
      print(img, "OK")
      gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      gray = cv2.GaussianBlur(gray, (5, 5), 0)
      # threshold the image, then perform a series of erosions +
      # dilations to remove any small regions of noise
      thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
      thresh = cv2.erode(thresh, None, iterations=2)
      thresh = cv2.dilate(thresh, None, iterations=2)
      # find contours in thresholded image, then grab the largest one
      cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      c = max(cnts, key=cv2.contourArea)
      # find the extreme points
      extLeft = tuple(c[c[:, :, 0].argmin()][0])
      extRight = tuple(c[c[:, :, 0].argmax()][0])
      extTop = tuple(c[c[:, :, 1].argmin()][0])
      extBot = tuple(c[c[:, :, 1].argmax()][0])
      ADD_PIXELS = 0
      new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
      return tf.convert_to_tensor(new_img)


def img_enhanced(img):
  print(type(img))
  img = img.numpy()
  print(img, "OK")

  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  gray = cv2.GaussianBlur(gray, (5, 5), 0)
  # threshold the image, then perform a series of erosions +
  # dilations to remove any small regions of noise
  thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
  thresh = cv2.erode(thresh, None, iterations=2)
  thresh = cv2.dilate(thresh, None, iterations=2)
  # find contours in thresholded image, then grab the largest one
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  c = max(cnts, key=cv2.contourArea)
  # find the extreme points
  extLeft = tuple(c[c[:, :, 0].argmin()][0])
  extRight = tuple(c[c[:, :, 0].argmax()][0])
  extTop = tuple(c[c[:, :, 1].argmin()][0])
  extBot = tuple(c[c[:, :, 1].argmax()][0])
  ADD_PIXELS = add_pixels_value
  new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
  return new_img

inputs = keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
# x = img_enhanced(x)
x = ImagePreprocessLayer()(x)
x = keras.layers.Normalization(mean=MEAN, variance=STD)(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = keras.Model(inputs, outputs)


class Model(tf.Module):
    def __init__(self):
        self.model = model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(from_logits=True))

    @tf.function(input_signature=[
        tf.TensorSpec([None, 160, 160, 3], tf.float32),
        tf.TensorSpec([None, ], tf.float32),
    ])
    def train(self, x, y):
      with tf.GradientTape() as tape:
          pred = self.model(x)
          loss = self.model.loss(y, pred)

      gradients = tape.gradient(loss, self.model.trainable_variables)
      self.model.optimizer.apply_gradients(
          zip(gradients, self.model.trainable_variables))
      result = {"loss": loss}
      return result

    @tf.function(input_signature=[
        tf.TensorSpec([None, 160, 160, 3], tf.float32)
    ])
    def infer(self, x):
        logits = self.model(x)
        probabilities = tf.nn.sigmoid(x)
        return {
            "output": probabilities,
            "logits": logits
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors
