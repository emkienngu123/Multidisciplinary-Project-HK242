import sys
from Adafruit_IO import MQTTClient
import time
import lancedb
import pyarrow as pa
import datetime
import pandas as pd
from env import *

feeds =  AIO_FEED_ID

lancedb_instance = lancedb.connect('database.lance')
TABLE_NAME = ['TEMPERATURE','HUMIDITY','LIGHT','MOVEMENT']
if 'TEMPERATURE' not in lancedb_instance.table_names():
    temperature_schema = pa.schema([
        pa.field('date',pa.string()),
        pa.field('value',pa.float32())
    ])

    lancedb_instance.create_table('TEMPERATURE',schema=temperature_schema)

if 'HUMIDITY' not in lancedb_instance.table_names():
    humidity_schema = pa.schema([
        pa.field('date',pa.string()),
        pa.field('value',pa.float32())
    ])

    lancedb_instance.create_table('HUMIDITY',schema=humidity_schema)

if 'LIGHT' not in lancedb_instance.table_names():
    light_schema = pa.schema([
        pa.field('date',pa.string()),
        pa.field('value',pa.float32())
    ])

    lancedb_instance.create_table('LIGHT',schema=light_schema)

if 'MOVEMENT' not in lancedb_instance.table_names():
    movement_schema = pa.schema([
        pa.field('date',pa.string()),
        pa.field('value',pa.int32())
    ])

    lancedb_instance.create_table('MOVEMENT',schema=movement_schema)




#PREPROCESS SENSOR DATA
def process_humidity_data(payload):
    df = pd.DataFrame(
    {
        'date' : [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'value' : float(payload)
    }
    )
    lancedb_instance['HUMIDITY'].add(df)

def process_light_data(payload):
    df = pd.DataFrame(
    {
        'date' : [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'value' : float(payload)
    }
    )
    lancedb_instance['LIGHT'].add(df)

def process_movement_data(payload):
    df = pd.DataFrame(
    {
        'date' : [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'value' : int(payload)
    }
    )
    lancedb_instance['MOVEMENT'].add(df)

def process_temperature_data(payload):
    df = pd.DataFrame(
    {
        'date' : [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'value' : float(payload)
    }
    )
    lancedb_instance['TEMPERATURE'].add(df)



def process_data(payload,feed_id):
    if feed_id == 'yolo-humidity-sensor':
        process_humidity_data(payload)
    elif feed_id == 'yolo_light_sensor':
        process_light_data(payload)
    elif feed_id == 'yolo-movement-sensor':
        process_movement_data(payload)
    elif feed_id == 'yolo-temperature-sensor':
        process_temperature_data(payload)





def connected(client):
    print('Connect successfully')
    for feed in feeds:
        client.subscribe(feed)

def subscribe(client,userdata,mid,granted_qos):
    print('Subscribe successfully')

def disconnected(client):
    print('Disconnecting')
    sys.exit(1)

def message(client, feed_id ,payload):
    print('Receive message: ' + payload+ ' from ' + feed_id)
    process_data(payload,feed_id)
    

client = MQTTClient(AIO_USERNAME,AIO_KEY)
client.on_connect = connected
client.on_disconnect = disconnected
client.on_message = message
client.on_subscribe = subscribe
client.connect()
client.loop_background()

try:
    while True:
        user_input = input("Enter data to send to the dashboard: ")
        # Publish the user input to the designated feed
        client.publish('yolo-fan-device', user_input)
        print(f"Sent: {user_input}")
        # Add a small delay if necessary
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
    client.disconnect()
