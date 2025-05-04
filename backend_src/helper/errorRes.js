class BadRequestError extends Error {
    constructor(message) {
        super(message);
        this.status = 400;
    }
}

class UnauthorizedError extends Error {
    constructor(message) {
        super(message);
        this.status = 401;
    }
}

class NotFoundError extends Error {
    constructor(message) {
        super(message);
        this.status = 404;
    }
}

class ForbiddenError extends Error {
    constructor(message) {
        super(message);
        this.status = 403;
    }
}

class InternalServerError extends Error {
    constructor(message) {
        super(message);
        this.status = 404;
    }
}

export {BadRequestError, UnauthorizedError, NotFoundError, ForbiddenError, InternalServerError}