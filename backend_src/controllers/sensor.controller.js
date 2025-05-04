import SensorService from "../services/sensor.service.js"
import path from 'path';
import fs from 'fs';
import FormData from 'form-data';
import axios from 'axios';
import { NotFoundError, BadRequestError } from "../helper/errorRes.js";


class SensorController {
    static async getSensor(req, res) {
        // from req.user.id to get all sensor
        console.log("Get sensor")
        res.status(200).send({
            message: "data about all sensors of user",
            data: [
                {id: "yolo_light_sensor",name: "light sensor"},
                {id: "yolo-humidity-sensor",name: "humid sensor"},
                {id: "yolo_temperature_sensor",name: "temperature sensor"},
                {id: "yolo_movement_sensor",name: "movement sensor"}
            ]
        })
    }

    static async getTemperature(req, res) {
        // from req.user.id to get all sensor

        // console.log("XX")
        console.log("Get temp")
        res.status(200).send({
            message: "Temperature data",
            data: await SensorService.getTemperature(req.user.id, req.query.numRecord)
        })
    }

    static async getHumidity(req, res) {
        // from req.user.id to get all sensor
        console.log("Get humidity")
        res.status(200).send({
            message: "Humidity data",
            data: await SensorService.getHumidity(req.user.id, req.query.numRecord)
        })
    }

    static async getLight(req, res) {
        // from req.user.id to get all sensor
        console.log("Get light")
        res.status(200).send({
            message: "Light data",
            data: await SensorService.getLight(req.user.id, req.query.numRecord)
        })
    }

    static async getMovement(req, res) {
        // from req.user.id to get all sensor
        console.log("Get movement")
        res.status(200).send({
            message: "Movement data",
            data: await SensorService.getMovement(req.user.id, req.query.numRecord)
        })
    }

    static async getFanSpeed(req, res) {
        // from req.user.id to get all sensor
        console.log("Get fan speed")
        res.status(200).send({
            message: "Fan speed data",
            data: await SensorService.getFanSpeed(req.user.id)
        })
    }

    static async getLightIntensity(req, res) {
        // from req.user.id to get all sensor
        console.log("Get light intensity")
        res.status(200).send({
            message: "Light data",
            data: await SensorService.getLightIntensity(req.user.id)
        })
    }

    static async setFanSpeed(req, res) {
        // from req.user.id to get all sensor
        console.log("Set fan speed", req.body.speed)
        await SensorService.setFanSpeed(req.body.speed, req.user.id)

        res.status(200).send({
            message: "Successfully",
            data: req.body.speed
        })
    }

    static async setLightIntensity(req, res) {
        // from req.user.id to get all sensor
        console.log("Set light", req.body.value)
        await SensorService.setLightIntensity(req.body.value, req.user.id)

        res.status(200).send({
            message: "Successfully",
            data: req.body.value
        })
    }

    static async voiceProcess(req, res) {

        console.log("Voice command")
        const file = req.file;

        if (!file || path.extname(file.originalname).toLowerCase() !== '.mp3') {
            return res.status(400).json({ error: 'Invalid MP3 file' });
        }

        const form = new FormData();
        form.append('audio', fs.createReadStream(file.path), file.originalname);

        try {
            const response = await axios.post(
                process.env.VOICE_INFERENCE_HOST + '/predict',
                form,
                { headers: form.getHeaders() }
            )

            console.log("Voice command result:", response.data)

            if (response.data.result == 'failed') {
                return res.status(400).json({ error: response.data.message});
            }
            else {
                if (response.data.message == "increase fan") {
                    let current_speed = await SensorService.getFanSpeed(req.user.id)
                    await SensorService.setFanSpeed(Math.max(0, current_speed - 40), req.user.id)
                }
                else if (response.data.message == "decrease fan") {
                    let current_speed = await SensorService.getFanSpeed(req.user.id)
                    await SensorService.setFanSpeed(Math.min(100, current_speed + 40), req.user.id)
                }
                else if (response.data.message == "decrease light") {
                    let current_light = await SensorService.getLightIntensity(req.user.id)
                    await SensorService.setLightIntensity(Math.max(0, current_light - 40), req.user.id)
                }
                else if (response.data.message == "increase light") {
                    let current_light = await SensorService.getLightIntensity(req.user.id)
                    await SensorService.setLightIntensity(Math.min(100, current_light + 40), req.user.id)
                }
                else {
                    throw new BadRequestError("Voice command is not recognized")
                }

                res.status(200).json({
                    message: "success",
                    data: response.data.message
                });
            }
        
            fs.unlink(file.path, () => {}); // optional: delete local file
        } catch (err) {
            if (err instanceof BadRequestError) {
                console.error('Voice command is not recognized');
                throw new BadRequestError("Voice command is not recognized")
            }
            else if(err instanceof NotFoundError) {
                console.error('IOT not found');
                throw new NotFoundError(err.message)
            }
            else {
                console.error('Error forwarding MP3:', err.message);
                res.status(500).json({ error: 'Voice processing failed' });
            }
        }
    }

    static async setAutoAdjust(req, res) {
        console.log("Set auto adjust")
        res.status(200).send({
            message: "Auto adjustment",
            data: await SensorService.setAutoAdjust(req.user.id, req.query.enable)
        })
    }

    static async getAlert(req, res) {
        console.log("Get alert")
        res.status(200).send({
            message: "Alert",
            data: await SensorService.getAlert(req.user.id)
        })
    }
}

export default SensorController;