from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import imutils
import easyocr
import os

app = Flask(__name__)
CORS(app)
vehicle_number = None
rendered_image_path = None

def recognize_number_plate(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection
    edged = cv2.Canny(bfilter, 30, 200)
    
    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    # Find location of number plate
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    # Create mask
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    
    
    # Crop the number plate region
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
    
    # Save cropped image
    output_image_path = 'output_image.jpg'
    cv2.imwrite(output_image_path, cropped_image)
    
    # Perform OCR on cropped image
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    
    # Extract alphanumeric characters
    vehicle_number = ''.join(filter(str.isalnum, result[0][-2].upper()))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img.copy(), text=vehicle_number, org=(location[0][0][0], location[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img.copy(), tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3)
    
    # Save rendered image
    rendered_image_path = 'rendered_image.jpg'
    cv2.imwrite(rendered_image_path, res)
    
    return vehicle_number, rendered_image_path


@app.route('/api/vehicle-details', methods=['POST'])
def get_vehicle_details():
    try:
        # # Receive image file
        # image_file = request.files['image']
        
        # # Save image temporarily
        # image_path = 'temp_image.jpg'
        # image_file.save(image_path)
        
        # # Recognize number plate
        # vehicle_number, rendered_image_path = recognize_number_plate(image_path)
        
        return jsonify({'vehicle_number': 'R22S9769'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True)
