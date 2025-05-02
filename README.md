create file env.py and declare your Adafruit AIO_FEED_ID,AIO_USERNAME and AIO_KEY  
```
python .\IoTGateWay.py : collect data from Adafruit Dashboard
```


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

train VoiceCommand model
```
cd AI/VoiceCommand
python train.py --cfg cfg/main.yaml > outputs/voicecommand.log
```
inference VoiceCommand model
```
cd AI/VoiceCommand
python inference.py --cfg cfg/main.yaml
```

train FACEREG
```
cd AI/FaceReg
python train.py --cfg cfg/main.yaml > outputs/facereg.log
```
inference FACEREG model

GENERATE VECTOR DATA
```
cd AI/FaceReg
python vector_generate.py --cfg cfg/main.yaml
```
INFERENCE
```
cd AI/FaceReg
python inference.py --cfg cfg/main.yaml
```