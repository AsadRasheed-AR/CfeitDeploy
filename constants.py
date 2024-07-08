import os

# Global Parameters
DATASET_BASEDIR = '/content/drive/MyDrive/Counterfeit/Dataset'
SIGREF_BASEDIRE = r"C:/Asad's Workspace/Counterfeit/ASSETS/Signature Reference"
IMAGE_UPLOAD = r"C:/Asad's Workspace/Counterfeit/ASSETS/upload"

NUM_PATCHES = 6  # Number of patches horizontally

SINGNATURE_TEMPLATES = [
    # Replace with your Reference Paths.
    os.path.join(SIGREF_BASEDIRE, "0.png"), # 0 : Ashraf Wathra
    os.path.join(SIGREF_BASEDIRE, "1.png"), # 1 : Raza Baqir
    os.path.join(SIGREF_BASEDIRE, "2.png"), # 2 : Jameel Ahmed.
    # os.path.join(SIGREF_BASEDIRE, "3.png"), # 3 : Raza Baqir.

    # Add more template paths as needed
]

SIGNATURE_DICT = {
    '0' : 'ASHRAF WATHRA',
    '1' : 'RAZA BAQIR',
    # '3' : 'RAZA BAQIR',
    '2' : 'JAMIL AHMED'
}

# Response Validations
ValidationResponse = {
        "RESP_TOP_SERIAL" : {
        "title": "Serial Top",
        "desc" : "The unique serial number printed at the top of the banknote, used for identification and authentication.",
        "status": False,
        "value": None
    },

    "RESP_BOTTOM_SERIAL": {
        "title": "Serial Bottom",
        "desc" : "The unique serial number printed at the bottom of the banknote, ensuring additional security and verification.",
        "status": False,
        "value": None
    },

    "RESP_PRINT_YEAR": {
        "title": "Note Printing Year",
        "desc" : "The year when the banknote was printed, providing information about the note's age and validity.",
        "status": False,
        "value": None
    },

    "RESP_DENOMINATION": {
        "title": "Denomination",
        "desc" : "The face value of the banknote, indicating its monetary value.",
        "status": False,
        "value": None
    },

    "RESP_GOVERNOR": {
        "title": "Governor SBP",
        "desc" : "The signature of the Governor of the State Bank of Pakistan, verifying the authenticity of the banknote.",
        "status": False,
        "value": None
    }
}

# # Response Validations
# PASS = {
#     "RESP_TOP_SERIAL" : {
#         "title": "Serial Top",
#         "desc" : "The unique serial number printed at the top of the banknote, used for identification and authentication.",
#         "status": True,
#         "value": None
#     },

#     "RESP_BOTTOM_SERIAL": {
#         "title": "Serial Bottom",
#         "desc" : "The unique serial number printed at the bottom of the banknote, ensuring additional security and verification.",
#         "status": True,
#         "value": None
#     },

#     "RESP_PRINT_YEAR": {
#         "title": "Note Printing Year",
#         "desc" : "The year when the banknote was printed, providing information about the note's age and validity.",
#         "status": True,
#         "value": None
#     },

#     "RESP_DENOMINATION": {
#         "title": "Denomination",
#         "desc" : "The face value of the banknote, indicating its monetary value.",
#         "status": True,
#         "value": None
#     },

#     "RESP_GOVERNOR": {
#         "title": "Governor SBP",
#         "desc" : "The signature of the Governor of the State Bank of Pakistan, verifying the authenticity of the banknote.",
#         "status": True,
#         "value": None
#     }
# }


# FAIL = {
#     "RESP_TOP_SERIAL" : {
#         "title": "Serial Top",
#         "desc" : "The unique serial number printed at the top of the banknote, used for identification and authentication.",
#         "status": False,
#         "value": None
#     },

#     "RESP_BOTTOM_SERIAL": {
#         "title": "Serial Bottom",
#         "desc" : "The unique serial number printed at the bottom of the banknote, ensuring additional security and verification.",
#         "status": False,
#         "value": None
#     },

#     "RESP_PRINT_YEAR": {
#         "title": "Note Printing Year",
#         "desc" : "The year when the banknote was printed, providing information about the note's age and validity.",
#         "status": False,
#         "value": None
#     },

#     "RESP_DENOMINATION": {
#         "title": "Denomination",
#         "desc" : "The face value of the banknote, indicating its monetary value.",
#         "status": False,
#         "value": None
#     },

#     "RESP_GOVERNOR": {
#         "title": "Governor SBP",
#         "desc" : "The signature of the Governor of the State Bank of Pakistan, verifying the authenticity of the banknote.",
#         "status": False,
#         "value": None
#     }
# }