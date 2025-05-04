import { clients } from "../shared/state.js";
import { NotFoundError, BadRequestError } from "../helper/errorRes.js";
import UserModel from "../models/user.model.js";

class SensorService {
    static async getTemperature(username, numRecord) {
        let ws = await UserModel.getAIO_ws(username)
        console.log(numRecord)
        return {
            date: ws.temperature_date.slice(-numRecord),
            value: ws.temperature.slice(-numRecord)
        } 
    }

    static async getHumidity(username, numRecord) {
        let ws = await UserModel.getAIO_ws(username)
        return {
            date: ws.humidity_date.slice(-numRecord),
            value: ws.humidity.slice(-numRecord)
        } 
    }

    static async getLight(username, numRecord) {
        let ws = await UserModel.getAIO_ws(username)
        return {
            date: ws.light_date.slice(-numRecord),
            value: ws.light.slice(-numRecord)
        } 
    }

    static async getMovement(username, numRecord) {
        let ws = await UserModel.getAIO_ws(username)
        return {
            date: ws.movement_date.slice(-numRecord),
            value: ws.movement.slice(-numRecord)
        } 
    }

    static async getFanSpeed(username) {
        let ws = await UserModel.getAIO_ws(username)
        return ws.fanspeed_data;
    }
    
    static async getLightIntensity(username) {
        let ws = await UserModel.getAIO_ws(username)
        return ws.light_data;
    }

    static async setFanSpeed(speed, username) {
        let ws = await UserModel.getAIO_ws(username)
        ws.fanspeed_data = speed 
        ws.send(JSON.stringify({device: 'fan', value: speed}))
    }

    static async setLightIntensity(brightness, username) {
        let ws = await UserModel.getAIO_ws(username)
        ws.light_data = brightness
        ws.send(JSON.stringify({device: 'light', value: brightness}))
    }

    static async setAutoAdjust(username, enable) {
        await UserModel.setAutoAdjust(username, enable)
        let ws = await UserModel.getAIO_ws(username)
        ws.auto_adjust = true
    }

    static async getAlert(username) {
        let ws = await UserModel.getAIO_ws(username)
        return ws.alert
    }
}

export default SensorService;