from dotenv import load_dotenv
import cx_Oracle
import oracledb
import os

from DataModel import CMDummyData

# Parse and load the enviornment variables.
load_dotenv(override=True)

# DB Configurations
USERNAME = os.environ.get("USERNAME")
PASSWORD = os.environ.get("PASSWORD")
DATABASE = os.environ.get("DB_NAME")
DB_PORT_IP = os.environ.get('DB_IP_PORT')
DB_SERVICE_NAME = os.environ.get('DB_ServiceName')
ENVIORNMENT = os.environ.get('ENVIORNMENT')
OCR = os.environ.get('OCR')

# Connection String
CONNECTION_STRING = f"{USERNAME}/{PASSWORD}@{DB_PORT_IP}/{DB_SERVICE_NAME}"

# DB Connect
DB_CONNECT = None

def connectToDB():
    global DB_CONNECT

    connection = cx_Oracle.connect(
    user=USERNAME,
    password=PASSWORD,
    dsn=f"{DB_PORT_IP}/{DB_SERVICE_NAME}")

    DB_CONNECT = connection
    return DB_CONNECT

def fetchCurrencyRecord(SerialTop,SerialBottom):
    global DB_CONNECT

    if ENVIORNMENT == 'DEV':
        record = filter_record_by_SNO(CMDummyData,SerialTop,SerialBottom)
        return record
    else:
        if DB_CONNECT is None:
            DB_CONNECT = connectToDB()
            print("Successfully connected to Oracle Database")

        with DB_CONNECT.cursor() as cursor:
            try:
                # Execute the query            
                sql = f"SELECT * FROM CM WHERE tserial = '{SerialTop}' or bserial = '{SerialBottom}'"
                record = cursor.execute(sql).fetchone()
                print(record)
                return record

            except oracledb.Error as e:
                error, = e.args
                print(sql)
                print('*'.rjust(error.offset+1, ' '))
                print(error.message)
                return error.message
            

# Function to filter records based on ID
def filter_record_by_SNO(records, TSNO, BSNO):
    for record in records:
        if record[1] == TSNO or record[2] == BSNO:
            return record
    return None