import sys
from Adafruit_IO import MQTTClient
import time
import lancedb
import pyarrow as pa
import datetime
import pandas as pd
from env import *
import random
import websocket
from websockets.sync.client import connect
import json
import threading

feeds =  AIO_FEED_ID
ws = connect("wss://smart-home-be-1.onrender.com")

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
    data = {
        'date' : [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'value' : float(payload)
    }
    
    df = pd.DataFrame(data)
    lancedb_instance['HUMIDITY'].add(df)
    
    data['aio_user'] = AIO_USERNAME 
    data['feed'] = 'yolo-humidity-sensor'
    ws.send(json.dumps(data))

def process_light_data(payload):
    data = {
        'date' : [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'value' : float(payload)
    } 
    
    df = pd.DataFrame(data)
    lancedb_instance['LIGHT'].add(df)
    
    data['aio_user'] = AIO_USERNAME 
    data['feed'] = 'yolo-light-sensor'
    ws.send(json.dumps(data))

def process_movement_data(payload):
    data = {
        'date' : [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'value' : int(payload)
    }
    
    df = pd.DataFrame(data)
    lancedb_instance['MOVEMENT'].add(df)
    
    data['aio_user'] = AIO_USERNAME 
    data['feed'] = 'yolo-movement-sensor'
    ws.send(json.dumps(data))

def process_temperature_data(payload):
    data = {
        'date' : [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'value' : float(payload)
    }
    
    df = pd.DataFrame(data)
    lancedb_instance['TEMPERATURE'].add(df)
    
    data['aio_user'] = AIO_USERNAME 
    data['feed'] = 'yolo-temperature-sensor'
    ws.send(json.dumps(data))


def process_data(payload,feed_id):
    if feed_id == 'yolo-humidity-sensor':
        process_humidity_data(payload)
    elif feed_id == 'yolo-light-sensor':
        process_light_data(payload)
    elif feed_id == 'yolo-movement-sensor':
        process_movement_data(payload)
    elif feed_id == 'yolo-temperature-sensor':
        process_temperature_data(payload)


def connected(client):
    print('Connect successfully')
    for feed in feeds:
        client.subscribe(feed)

def subscribe(client, userdata, mid, granted_qos):
    print('Subscribe successfully')

def disconnected(client):
    print('Disconnecting')
    sys.exit(1)

def message(client, feed_id ,payload):
    print('Receive message: ' + payload+ ' from ' + feed_id)
    process_data(payload, feed_id)

def fake_mess_recv():
    message(None, feed_id='yolo-temperature-sensor', payload=str(random.randint(30,34)))
    message(None, feed_id='yolo-humidity-sensor', payload=str(random.randint(60,70)))
    message(None, feed_id='yolo-light-sensor', payload=str(random.randint(95,100)))
    message(None, feed_id='yolo-movement-sensor', payload=str(random.randint(0,0)))
    time.sleep(1500)

client = MQTTClient(AIO_USERNAME,AIO_KEY)
client.on_connect = connected
client.on_disconnect = disconnected
client.on_message = message
client.on_subscribe = subscribe
client.connect()
client.loop_background()

def websocket_listener():
    while True:
        try:
            message = ws.recv()
            data = json.loads(message)
            if data['device'] == 'fan':
                client.publish('yolo-fan-device', data['value'])
                print(f"Sent fan speed: {data['value']}")
            elif data['device'] == 'light':
                client.publish('yolo-light-led', data['value'])
                print(f'Sent light intensity: {data['value']}')
            else:
                pass
                        
        except Exception as e:
            print(f"WebSocket error: {e}")
            break
        
threading.Thread(target=websocket_listener, daemon=True).start()

while True:
    fake_mess_recv()
    # pass