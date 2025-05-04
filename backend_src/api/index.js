import express from 'express';
import dotenv from 'dotenv';
import pool from '../dbs/index.js';
import { asyncHandler } from '../utils/index.js';
import AccessController from '../controllers/access.controller.js';
import SensorController from '../controllers/sensor.controller.js';
import { NotFoundError } from '../helper/errorRes.js';
import cors from "cors";
import cookieParser from "cookie-parser";
import authMiddleware from "../middlewares/auth.middle.js"
import http from 'http';
import WebSocket, { WebSocketServer } from 'ws';
import axios from 'axios'
import { clients } from '../shared/state.js'
import multer from 'multer';

dotenv.config();

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });
const upload = multer({ dest: 'uploads/' });
// const clients = new Map(); // aio_id -> websocket_from_gateway

wss.on('connection', (ws) => {
    console.log('IOTGateWay worker connected via WebSocket');

    ws.on('message', (message) => {
        const data = JSON.parse(message);
        
        const aio_user = data['aio_user']
        if (!clients.has(aio_user)) {
            ws.aio_user = aio_user
            ws.count = [0, 0, 0, 0]
            ws.humidity = []
            ws.light = []
            ws.movement = []
            ws.temperature = []
            ws.humidity_date = []
            ws.light_date = []
            ws.movement_date = []
            ws.temperature_date = []
            ws.light_data = 0
            ws.fanspeed_data = 0
            ws.auto_adjust = true
            ws.alert = []
            clients.set(aio_user, ws);
        }

        if (data['feed'].includes('humidity')) {
            ws.humidity.push(data['value'])
            ws.humidity_date.push((new Date()).toISOString())
            ws.count[0] += 1
        }
        else if (data['feed'].includes('light')) {
            ws.light.push(data['value'])
            ws.light_date.push((new Date()).toISOString())
            ws.count[1] += 1
        }
        else if (data['feed'].includes('temperature')) {
            ws.temperature.push(data['value'])
            ws.temperature_date.push((new Date()).toISOString())
            ws.count[2] += 1
        }
        else if (data['feed'].includes('movement')) {
            ws.movement.push(data['value'])
            ws.movement_date.push((new Date()).toISOString())
            ws.count[3] += 1
        }

        if(ws.count[0] >= 5 && ws.count[1] >= 5 && ws.count[2] >= 5 && ws.count[3] >= 5) {
            axios.post(process.env.SMARTHOME_INFERENCE_HOST + '/predict', {
                temperature: ws.temperature.slice(-5),
                light: ws.light.slice(-5),
                humidity: ws.humidity.slice(-5),
                movement: ws.movement.slice(-5),
            })
            .then(response => {
                console.log('Response:', response.data);
                if (response.data.anomaly == 0 && ws.auto_adjust) {
                    ws.send(JSON.stringify({device: 'fan', value: response.data.fan_preds}))
                    ws.fanspeed_data = response.data.fan_preds
                    ws.send(JSON.stringify({device: 'light', value: response.data.light_preds}))
                    ws.light_data = response.data.light_preds
                }
                else if(response.data.anomaly != 0) {
                    ws.alert.push({date: (new Date()).toISOString(), message: "Parameters show that your house is showing signs of abnormality."})
                }
            })
            .catch(error => {
                console.error('Error:', error.message);
            });

            ws.count = [0, 0, 0, 0]
        }
    });
  
    ws.on('close', () => {
        clients.delete(ws.aio_user)
        console.log('IOTGateWay worker disconnected');
    });
});

app.use(express.json());
app.use(
    cors({
        origin: ["https://a.com", "http://localhost:3000", "http://127.0.0.1"], // Only allow your frontend
        credentials: true
    })
);

app.use(cookieParser());

app.get("/", (req, res) => res.send("Express on Vercel"));

app.post("/api/signup", asyncHandler(AccessController.signUp))
app.post("/api/signin", asyncHandler(AccessController.signIn))
app.post("/api/facesignin", upload.single('image'), asyncHandler(AccessController.signInByFace))

app.use(asyncHandler(authMiddleware))    

app.get("/api/sensor", asyncHandler(SensorController.getSensor));

app.get("/api/temperature", asyncHandler(SensorController.getTemperature));
app.get("/api/humidity", asyncHandler(SensorController.getHumidity));
app.get("/api/light", asyncHandler(SensorController.getLight));
app.get("/api/movement", asyncHandler(SensorController.getMovement));
app.get("/api/fanspeed", asyncHandler(SensorController.getFanSpeed));
app.get("/api/lightintensity", asyncHandler(SensorController.getLightIntensity));
app.get("/api/alert", asyncHandler(SensorController.getAlert));

// update fan speed
app.patch("/api/autoadjust", asyncHandler(SensorController.setAutoAdjust))
app.patch("/api/fanspeed", asyncHandler(SensorController.setFanSpeed))
app.patch("/api/lightintensity", asyncHandler(SensorController.setLightIntensity))

app.post("/api/voicecommand", upload.single('audio'), asyncHandler(SensorController.voiceProcess))

app.post("/api/test", asyncHandler(async (req, res) => {
    console.log(req.user)
    res.status(200).send({
        message: "Test",
        data: 1
    })
}));

app.use((req, res, next) => {
    const error = new NotFoundError("Not found this route!");
    error.status = 404;
    next(error);
});

app.use((error, req, res, next) => {
    console.log("Error::", error.message);
    res.status(error.status || 500);
    res.json({
        status: "error",
        message: error.message
    });
});

const PORT = 3000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));