# from constants import ValidationResponse
from constants import ValidationResponse

def SerialTop_Validation(SerialNum, CMSerialNum):
    ResponseTopSerial = ValidationResponse['RESP_TOP_SERIAL']
    ResponseTopSerial['value'] = SerialNum
    ResponseTopSerial['status'] = False


    if SerialNum == CMSerialNum:
        ResponseTopSerial['status'] = True
        return ResponseTopSerial
    else:
        return ResponseTopSerial

def SerialBottom_Validation(SerialNum, CMSerialNum):
    ResponseBottomSerial = ValidationResponse['RESP_BOTTOM_SERIAL']
    ResponseBottomSerial['value'] = SerialNum
    ResponseBottomSerial['status'] = False


    if SerialNum == CMSerialNum:
        ResponseBottomSerial['status'] = True
        return ResponseBottomSerial
    else:
        return ResponseBottomSerial

def PrintYear_Validation(PrintYear, CMPrintYear):
    ResponsePrintYear = ValidationResponse['RESP_PRINT_YEAR']
    ResponsePrintYear['value'] = PrintYear
    ResponsePrintYear['status'] = False


    if PrintYear == CMPrintYear:
        ResponsePrintYear['status'] = True
        return ResponsePrintYear
    else:
        return ResponsePrintYear

def Denomination_Validation(Denomination, CMDenomination):
    ResponseDenomination = ValidationResponse['RESP_DENOMINATION']
    ResponseDenomination['value'] = Denomination
    ResponseDenomination['status'] = False


    if Denomination == CMDenomination:
        ResponseDenomination['status'] = True
        return ResponseDenomination
    else:
        return ResponseDenomination

def Governor_Validation(Governor,CMGovernor):
    ResponseGovernor = ValidationResponse['RESP_GOVERNOR']
    ResponseGovernor['value'] = Governor
    ResponseGovernor['status'] = False


    if Governor == CMGovernor:
        ResponseGovernor['status'] = True
        return ResponseGovernor
    else:
        return ResponseGovernor

