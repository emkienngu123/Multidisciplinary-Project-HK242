import AccessService from "../services/access.service.js";

class AccessController {
    static async signUp(req, res) {

        console.log("Sign up::", { "username": req.body.username, "pass": req.body.password } )

        const result = await AccessService.signUp(req.body.username, req.body.password);
        res.status(200);
        res.send({
            message: "Sign up successfully!",
            data: result
        })
    }

    static async signIn(req, res) {
        console.log("Sign in::", { "username": req.body.username, "pass": req.body.password } )
        const result = await AccessService.signIn(req.body.username, req.body.password);
        res.cookie("token", result.accessToken, {httpOnly: true, sameSite: "none"});
        res.status(200);
        res.send({
            message: "Sign in successfully!",
            data: result
        })
    }

    static async signInByFace(req, res) {
        console.log("Sign in by face" )
        const result = await AccessService.signInByFace(req);
        res.cookie("token", result.accessToken, {httpOnly: true, sameSite: "none"});
        res.status(200);
        res.send({
            message: "Sign in successfully!",
            data: result
        })
    }
}

export default AccessController;