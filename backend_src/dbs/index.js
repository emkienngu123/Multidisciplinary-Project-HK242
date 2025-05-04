import pkg from 'pg';
const { Pool } = pkg;

import dotenv from 'dotenv';
dotenv.config();

// Create a new PostgreSQL connection pool
const pool = new Pool({
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    ssl: {
        rejectUnauthorized: false 
    }
});

// Test the database connection
pool.connect()
    .then(client => {
        console.log('Connected to PostgreSQL');
        client.release(); // Release the client back to the pool
    })
    .catch(err => console.error('PostgreSQL Connection Error:', err));

export default pool;