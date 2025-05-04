import express from 'express';
import dotenv from 'dotenv';
import pool from './dbs/index.js';
import { asyncHandler } from './utils/index.js';
import AccessController from './controllers/access.controller.js';
import SensorController from './controllers/sensor.controller.js';
import { NotFoundError } from './helper/errorRes.js';
import cors from "cors";
import cookieParser from "cookie-parser";
import authMiddleware from "./middlewares/auth.middle.js"
dotenv.config();

const app = express();
app.use(express.json());
// app.use(
//     cors({
//         origin: ["https://a.com", "http://localhost:3000"], // Only allow your frontend
//         credentials: true
//     })
// );
// currently accept all
app.use(
    cors({
        origin: (origin, callback) => {
            callback(null, origin || "*"); // Chấp nhận tất cả
        },
        credentials: true
    })
);
app.use(cookieParser());

app.post("/api/signup", asyncHandler(AccessController.signUp))
app.post("/api/signin", asyncHandler(AccessController.signIn))

app.use(asyncHandler(authMiddleware))

app.get("/api/sensor", asyncHandler(SensorController.getSensor));

app.get("/api/temperature", asyncHandler(SensorController.getTemperature));
app.get("/api/humidity", asyncHandler(SensorController.getHumidity));
app.get("/api/light", asyncHandler(SensorController.getLight));
app.get("/api/movement", asyncHandler(SensorController.getMovement));
app.get("/api/fanspeed", asyncHandler(SensorController.getFanSpeed));

// update fan speed
app.patch("/api/fanspeed", asyncHandler(SensorController.setFanSpeed))

app.post("/api/test", asyncHandler(async (req, res) => {
    console.log(req.user)
    res.status(200).send({
        message: "Test",
        data: 1
    })
}));

app.use((req, res, next) => {
    const error = new NotFoundError("Not found! ");
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

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));