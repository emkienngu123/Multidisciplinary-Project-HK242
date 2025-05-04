import db from "../dbs/index.js"
import { clients } from "../shared/state.js";
import { NotFoundError, BadRequestError } from "../helper/errorRes.js";

class UserModel {
    static async findUser(username) {
        const res = await db.query("SELECT * FROM users WHERE username = $1", [username]);
        return res.rows
    }

    static async newUser(username, password) {
        await db.query("INSERT INTO users (username, password) VALUES ($1, $2)", [username, password])
    }

    static async getAIO(username) {
        const res = await db.query("SELECT users.aio_user as aio_user, aio_key FROM users LEFT JOIN aio_users ON aio_users.aio_user = users.aio_user WHERE username = $1", [username]);
        
        if (res.rows.length == 0) {
            throw new BadRequestError("User name not found")
        }

        if (typeof res.rows[0].aio_user == 'undefined') {
            throw new BadRequestError("User does not have any IOT Gateway")
        }

        // console.log(res.rows[0])
        return res.rows[0]
    }

    static async getAIO_ws(username) {
        const res = await db.query("SELECT aio_user FROM users WHERE username = $1", [username]);
        
        if (res.rows.length == 0) {
            throw new BadRequestError("User name not found")
        }

        if (typeof res.rows[0].aio_user == 'undefined') {
            throw new BadRequestError("User does not have any IOT Gateway")
        }

        let ws = clients.get(res.rows[0].aio_user)

        if (typeof ws === 'undefined') {
            throw new NotFoundError("Your IOT Gateway is not connected")
        }

        return ws
    }

    static async setAutoAdjust(username, enable) {
        const res = await db.query("SELECT aio_user FROM users WHERE username = $1", [username]);
        
        if (res.rows.length == 0) {
            throw new BadRequestError("User name not found")
        }

        if (enable == "true") { db.query("UPDATE aio_users SET auto_adjust = TRUE WHERE aio_user = $1", [res.rows[0].aio_user]) }
        else { db.query("UPDATE aio_users SET auto_adjust = FALSE WHERE aio_user = $1", [res.rows[0].aio_user]) }
    }

    static async getUserByFaceId(id) {
        const res = await db.query("SELECT username FROM users WHERE face_id = $1", [id]);

        if (res.rows.length == 0) {
            throw new BadRequestError("Face not found")
        }

        // console.log(res.rows)

        return res.rows[0]
    }
}

export default UserModel;