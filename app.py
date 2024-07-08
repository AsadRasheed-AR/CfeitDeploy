from flask import Flask, request
import os

import numpy as np

from constants import NUM_PATCHES, SINGNATURE_TEMPLATES, SIGNATURE_DICT, IMAGE_UPLOAD
from utils import split_and_plot_image, extract_serial, extract_print_year, extract_denomination, extract_signature
# from constants import PASS,FAIL
from validations import SerialTop_Validation, SerialBottom_Validation, Denomination_Validation, PrintYear_Validation, Governor_Validation
from CMValidations import fetchCurrencyRecord
import uuid

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Conterfeit Test API.</p>"

# Test Route
@app.route("/test")
def test_route():
    return "API working!"

@app.route("/verify", methods=['POST'])
def verify():
    image = request.files.get("img")

    if image:
        # Generate a unique filename
        unique_filename = str(uuid.uuid4()) + os.path.splitext(image.filename)[1]
        
        # Define the full path
        filepath = os.path.join(IMAGE_UPLOAD, unique_filename)
        
        # Save the file
        image.save(filepath)

    # Example usage:
    img_parts = split_and_plot_image(image, NUM_PATCHES, show_images=False)

    """Extract and Show Features"""

    Serial_top = extract_serial(np.hstack((img_parts[4], img_parts[5])))
    Serial_Bottom = extract_serial(np.hstack((img_parts[12], img_parts[13])))
    print_year = extract_print_year(img_parts[11])
    denomination = extract_denomination(np.hstack((img_parts[16], img_parts[17])))
    Governor_name = extract_signature(np.hstack((img_parts[7], img_parts[8])), SINGNATURE_TEMPLATES, SIGNATURE_DICT, show_image= False)
    # Governor_name = extract_signature(img_parts[8], SINGNATURE_TEMPLATES, SIGNATURE_DICT, show_image= False)

    # Serial_top, Serial_Bottom, print_year, denomination, Governor_name

    Serial_top = Serial_top.upper() if Serial_top != None else Serial_top
    Serial_Bottom = Serial_Bottom.upper() if Serial_Bottom != None else Serial_Bottom
    Governor_name = Governor_name.upper() if Governor_name != None else Governor_name

    print(f"Top Serial Number = {Serial_top}")
    print(f"Bottom Serial Number = {Serial_Bottom}")
    print(f"Printing Year = {print_year}")
    print(f"Denomination = {denomination}")
    print(f"Governor SBP = {Governor_name}")

    if Serial_top == None or Serial_Bottom == None:
        return {
            "response" : "Please provide a clear picture of the bank note."
        }
    else:
        # Features = [
        #         [PASS['RESP_TOP_SERIAL'] if Serial_top.lower() != None else FAIL['RESP_TOP_SERIAL']],
        #         [PASS['RESP_BOTTOM_SERIAL'] if Serial_Bottom.lower() != None else FAIL['RESP_BOTTOM_SERIAL']],
        #         [PASS['RESP_PRINT_YEAR'] if print_year != None else FAIL['RESP_PRINT_YEAR']],
        #         [PASS['RESP_DENOMINATION'] if denomination != None else FAIL['RESP_DENOMINATION']],
        #         [PASS['RESP_GOVERNOR'] if (Governor_name != None and Governor_name != 'No signatures found.') else FAIL['RESP_GOVERNOR']],
        #     ]
        record = fetchCurrencyRecord(Serial_top.upper(), Serial_Bottom.upper())

        if record is None:
            Features = [
                    SerialTop_Validation(Serial_top, ''),
                    SerialBottom_Validation(Serial_Bottom, ''),
                    PrintYear_Validation(print_year, ''),
                    Denomination_Validation(denomination, ''),
                    Governor_Validation(Governor_name, '')
                ]
        else:
            count = 0
            Features = [
                    SerialTop_Validation(Serial_top, record[1].strip()),
                    SerialBottom_Validation(Serial_Bottom, record[2].strip()),
                    PrintYear_Validation(print_year, record[3].strip()),
                    Denomination_Validation(denomination, record[4].strip()),
                    Governor_Validation(Governor_name, record[5].strip())
                ]
        return {
            "Features": Features,
            "Score_Total" : len(Features),
            "Score": sum(1 for feature in Features if feature['status'] is not False)
        }
        # return {
        #     "Top Serial" : Serial_top.upper(),
        #     "Bottom Serial": Serial_Bottom.upper(),
        #     "Print Year": print_year,
        #     "Denomination": denomination,
        #     "Governor Name": Governor_name
        # }

if __name__ == '__main__':   
    app.run(debug=True)