import jwt from "jsonwebtoken";

const authMiddleware = (req, res, next) => {
    // Get token from headers
    let token = req.cookies?.token; // Bearer <token>

    // console.log("Authentication...")

    if (!token) {
        // console.log("No token provided via cookie")
        let authHeader = req.headers["authorization"];
        let token_from_req = authHeader && authHeader.split(" ")[1];
        if(!token_from_req) {
            // console.log("No token provided in header")
            return res.status(401).json({ message: "Unauthorized: No token provided" });
        }
        token = token_from_req;
    }

    try {
        // Verify token
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        req.user = decoded; // Attach user data to request
        console.log("User::",req.user)
        next(); // Proceed to the next middleware or route
    } catch (error) {
        res.status(401).json({ message: "Unauthorized: Invalid token" });
    }
};

export default authMiddleware;