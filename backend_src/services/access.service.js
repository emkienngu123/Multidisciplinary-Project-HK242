import argon2 from "argon2";
import { BadRequestError } from "../helper/errorRes.js";
import userModel from "../models/user.model.js";
import jwt from "jsonwebtoken";
import dotenv from 'dotenv';
import UserModel from "../models/user.model.js";
import path from 'path';
import fs from 'fs';
import FormData from 'form-data';
import axios from 'axios';
dotenv.config();
// argon2.hash(password, { type: argon2.argon2id });
// argon2.verify(hash, password);

class AccessService {
    static async signUp(username, password) {
        if (!username || !password) {
            throw new BadRequestError("User name or password is not provided!")
        }

        const usernameRegex = /^[a-zA-Z0-9_]+$/;
        if (!usernameRegex.test(username)) {
            throw new BadRequestError("Username can only contain letter and number");
        }

        const userSearch = await userModel.findUser(username);
        if (userSearch.length > 0) {
            throw new BadRequestError("Username is exist!");
        }

        const hashedPass = await argon2.hash(password, { type: argon2.argon2id });
        await userModel.newUser(username, hashedPass);
        return {
            username: username
        }
    }

    static async signIn(username, password) {
        if (!username || !password) {
            throw new BadRequestError("User name or password is not provided!")
        }

        const userSearch = await userModel.findUser(username);
        if (userSearch.length == 0) {
            throw new BadRequestError("Username is not exist?");
        }

        const user = userSearch[0]
        const verify_pass = await argon2.verify(user.password, password);
        if (!verify_pass) {
            throw new BadRequestError("Wrong password!");
        }

        const accessToken = jwt.sign({ id: username }, process.env.JWT_SECRET || "secret", { expiresIn: "1h" });
        const aio = await UserModel.getAIO(username)

        return {
            username: username,
            accessToken: accessToken,
            aio_user: aio.aio_user,
            aio_key: aio.aio_key
        }
    }

    static async signInByFace(req) {
        const file = req.file;

        // console.log(req.file);

        if (!file || (path.extname(file.originalname).toLowerCase() !== '.png' && path.extname(file.originalname).toLowerCase() !== '.jpg')) {
            throw new BadRequestError('Invalid png/jpg file')
        }

        const form = new FormData();
        form.append('image', fs.createReadStream(file.path), file.originalname);

        try {
            const response = await axios.post(
                process.env.FACE_INFERENCE_HOST + '/predict',
                form,
                { headers: form.getHeaders() }
            )

            console.log("Face reg result:", response.data)

            if (response.data.result == 'failed') {
                throw new BadRequestError(response.data.messag)
            }
            else {
                if (response.data.id == -1) {
                    throw new BadRequestError("Face is not recognized")
                }

                const user = await UserModel.getUserByFaceId(response.data.id)
                const username = user.username

                const accessToken = jwt.sign({ id: username }, process.env.JWT_SECRET || "secret", { expiresIn: "1h" });
                const aio = await UserModel.getAIO(username)

                return {
                    username: username,
                    accessToken: accessToken,
                    aio_user: aio.aio_user,
                    aio_key: aio.aio_key
                }
            }
        
            // fs.unlink(file.path, () => {}); // optional: delete local file
        } catch (err) {
            if (err instanceof BadRequestError) {
                console.error('Face is not recognized');
                throw new BadRequestError("Face is not recognized", err.message)
            }
            else {
                console.error('Error forwarding image:', err.message);
                throw new BadRequestError("Face reg failed", err.message);
            }
        }
    }
}

export default AccessService;