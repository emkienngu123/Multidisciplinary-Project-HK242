create file env.py and declare your Adafruit AIO_FEED_ID,AIO_USERNAME and AIO_KEY  
python .\IoTGateWay.py : collect data from Adafruit Dashboard


train AI model

cd AI

python train.py --cfg cfg/main.yaml > outputs/smarthome.log