import cv2
import streamlit as st
import tensorflow as tf
import os
import numpy as np
import PIL
from CHECK_SSIM import show_growth_page
from predict_page import show_predict_page
from diff import show_diff_page
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
## Page Title
#st.set_page_config(page_title = "Cats vs Dogs Image Classification")
st.title("PET APP")
st.markdown("------")
page = st.sidebar.selectbox("SELECT", ("GROWTH","AGE PREDICTION","BACKGROUND REMOVER"))

if page == "GROWTH":
    show_growth_page()
elif page =="AGE PREDICTION":
    show_predict_page()
else:
    show_diff_page()

#url = "https://chaitu364590-checkssim-app7-epno5m.streamlit.app/"
#st.write("check out this [GROWTH FINDER](%s)" % url)
model_path="pety.tflite"

df = pd.read_csv("mlpa_p.csv")

y = df['AGE'].values.reshape(-1, 1)
X = df['RIGHT_EYE_SIZE'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
def calc(slope, intercept, Area):
    return slope*Area+intercept

score = calc(regressor.coef_, regressor.intercept_, 9.5)

score = regressor.predict([[9.5]])

y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')


# Load the labels into a list
classes = ['lefteye','righteye','nose','left eye','right eye']
#label_map = model.model_spec.config.label_map
#for label_id, label_name in label_map.as_dict().items():
 # classes[label_id-1] = label_name

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  return resized_img, original_image


def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Retur the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  scores = get_output_tensor(interpreter, 0)
  boxes = get_output_tensor(interpreter, 1)
  count = int(get_output_tensor(interpreter, 2))
  classes = get_output_tensor(interpreter, 3)

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
      
  return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.3):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path, 
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)
  
  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute 
    # coordinates based on the original image resolution
    #x,y,w,h = cv2.boundingRect(cnt)
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])
    w=xmax-xmin
    h=ymax-ymin
    
    #st.write(AreaofRectangle)
    # Find the class index of the current object
    class_id = int(obj['class_id'])
    
    st.write(classes[class_id])
    #st.write(class_id)
    if class_id!=2:
        Area = w * h
        Area=Area/240
        st.write("Area of a EYE is: %.2f" %Area)
        W = np.array([[Area]])
        W = W.astype(float)
        
        AGE = regressor.predict(W)
        st.subheader("The estimated AGE is :")
        st.write(AGE/16)
        st.markdown("__________________________")

    # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)  
  return original_uint8


## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png", 'jpeg'])
if uploaded_file is not None:
    with open(os.path.join("/tmp",uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())
    path = os.path.join("/tmp",uploaded_file.name)
    URL =path
    DETECTION_THRESHOLD = 0.3

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Run inference and draw detection result on the local copy of the original file
    detection_result_image = run_odt_and_draw_results(
        URL, 
        interpreter, 
        threshold=DETECTION_THRESHOLD
    )

    # Show the detection result
    st.image(detection_result_image)
