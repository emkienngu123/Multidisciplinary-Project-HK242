create file env.py and declare your Adafruit AIO_FEED_ID,AIO_USERNAME and AIO_KEY  
python .\IoTGateWay.py : collect data from Adafruit Dashboard


train SmartHome model
```
cd AI/SmartHome
python train.py --cfg cfg/main.yaml > outputs/smarthome.log
```
inference SmartHome model
```
cd AI/SmartHome
python inference.py --cfg cfg/main.yaml
```