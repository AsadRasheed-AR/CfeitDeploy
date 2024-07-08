import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import re

from PIL import Image
import cv2
# from google.colab.patches import cv2_imshow

from CMValidations import OCR

import operator

if OCR == 'EASYOCR':
  import easyocr
  reader = easyocr.Reader(['en'],model_storage_directory= './EASYOCR/',download_enabled=False) # this needs to run only once to load the model into memory
else:
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = 'C:/Tesseract-OCR/tesseract.exe'

# Utility Functions (Image Processing)

# Split an Image into Equal Parts
def split_and_plot_image(image_path, x, show_images= True):
    patches = []
    # Load image and convert to numpy array
    img = Image.open(image_path)
    img_array = np.array(img)

    # Determine dimensions
    height, width, channels = img_array.shape

    # Calculate number of patches vertically (assuming square patches)
    y = int(np.ceil(height / width * x))

    # Calculate patch dimensions
    patch_width = width // x
    patch_height = height // y

    # Create figure and subplots
    if show_images:
      fig = plt.figure(figsize=(10, 8))
      gs = gridspec.GridSpec(y, x, width_ratios=[1]*x, height_ratios=[1]*y)

    # Plot each patch
    for i in range(y):
        for j in range(x):
            patch = img_array[i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width, :]
            patches.append(patch)
            if show_images:
              ax = plt.subplot(gs[i, j])
              ax.imshow(patch)
              ax.set_xticks([])
              ax.set_yticks([])
    if show_images:
      plt.tight_layout()
      plt.show()
    return patches

# Feature Extraction
def extract_serial(image):
  # Extract Serial Number
  serial = None

  if OCR == 'EASYOCR':
    result = reader.readtext(image)
    for (bbox, text, prob) in result:
        # print(f'Text: {text}, Probability: {prob}')
        if len(text) > 6 and prob > 0.5:
          serial = text
  else:
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
     serial = pytesseract.image_to_string(thresh, lang='eng',config='--psm 6')

  return serial

def extract_print_year(image):
  # Extract Printing Year
  print_year = None

  if OCR == 'EASYOCR':
    result = reader.readtext(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    for (bbox, text, prob) in result:
        # print(f'Text: {text}, Probability: {prob}')
        if len(text) > 3 and prob > 0.7:
          print_year = text
  else:
     print_year = pytesseract.image_to_string(image)

  return print_year

def extract_denomination(image):
  # Extract Serial Number
  denom = None

  if OCR == 'EASYOCR':
    result = reader.readtext(image)
    for (bbox, text, prob) in result:
      if '5000' in text:
        denom = '5000'
      else:
        denom = re.findall(r'\d+',text)[0]
  else:
    text = pytesseract.image_to_string(image)
    if '5000' in text:
      denom = '5000'
    else:
      denom = re.findall(r'\d+',text)

  return denom

def detect_signature(image, template_paths):
    # Load the main image
    # main_image = image
    main_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # if main_gray.any():
    #     import uuid
    #     import os
    #         # Generate a unique filename
    #     unique_filename = str(uuid.uuid4())
        
    #     # Define the full path
    #     filepath = os.path.join(r'./ASSETS/Signature Reference', unique_filename + '.png')
        
    #     # Save the file
    #     cv2.imwrite(filepath,main_gray)
        # main_gray.save(filepath)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors of the main image
    keypoints_main, descriptors_query = sift.detectAndCompute(main_gray, None)

    # List to store results
    results = []

    # Iterate over each template
    for template_path in template_paths:
        # Load the template image
        template = cv2.imread(template_path, 0)  # Load as grayscale

        # Find keypoints and descriptors of the template
        keypoints_template, descriptors_train = sift.detectAndCompute(template, None)

        # Match descriptors between main image and template
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors_query, descriptors_train, k=2)
       
        # FLANN parameters
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks = 50)

        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(descriptors_main, descriptors_template, k=2)

        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good_matches.append(m)

        # If enough good matches are found, consider it a match
        if len(good_matches) > 3:  # Adjust this threshold as needed
            # Get the keypoints of the matches
            src_pts = np.float32([keypoints_main[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_template[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculate the homography matrix
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Get the corners of the template in the main image
            # h, w = template.shape
            # template_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # transformed_corners = cv2.perspectiveTransform(template_corners, M)

            # Store the result
            results.append({
                'template_path': template_path,
                # 'corners': transformed_corners,
                'score': len(good_matches)
            })

    return results

def extract_signature(image, template_paths, SIGNATURE_DICT, show_image= False):
  # Detect signatures
  results = detect_signature(image, template_paths)
  results.sort(key=operator.itemgetter('score'), reverse=True)

  # Visualize results (draw bounding boxes around detected signatures)
  # For testing Purpose.
  if results:
      main_image = image
      for result in results:
          # corners = result['corners']
          # cv2.polylines(image, [np.int32(corners)], True, (0, 255, 0), 2)
          break
      # Show the annotated image
      # k = result['template_path'].split('.')[0]
      k = re.findall(r'\d+', result['template_path'])[0]
      # print(f"Signature = {SIGNATURE_DICT[k]}")
      if show_image:
        cv2.imshow("Signature Extract", main_image)
      return SIGNATURE_DICT[k]
  else:
      return "No signatures found."

