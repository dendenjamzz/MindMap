const express = require('express');
const path = require('path');
const nodemailer = require('nodemailer');
const dotenv = require('dotenv');
const bcrypt = require('bcryptjs');
const mysql = require('mysql2');
const cors = require('cors');
const axios = require('axios');
const rateLimit = require('express-rate-limit');
const jwt = require('jsonwebtoken');
const { body, validationResult } = require('express-validator');

dotenv.config();

//allows requests from any origin (needed so the browser doesn't block the frontend)
//server to other server communication is allowed here
//only for development 
const corsOptions = {
    origin: (origin, callback) => {
        return callback(null, true);//allow all origins
    },
    methods: [
        'GET',     //read / fetch data
        'POST',    //create something new
        'PUT',     //update / replace something
        'DELETE',  //delete something
        'OPTIONS', //preflight request for CORS
    ],
    allowedHeaders: ['Content-Type', 'Accept', 'Authorization'],
};

const app = express();//for http responses
const PORT = 3002;

//logs every req for debug
app.use((req, res, next) => {
    console.log(`[${new Date().toLocaleString()}] ${req.method} ${req.path}`);
    next();
});

//rate limiter max 100 requests per 15 minutes per user, prevents spam/bots
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000,//window of 15 minutes in milisec
    max: 100,//100 requests max per window per IP
    standardHeaders: true, //how many requests are left and when the limit resets
    legacyHeaders: false,  //disables the old duplicate version of headers
});
app.use(limiter);

//secret key used to sign and verify JWT tokens
const JWT_SECRET = process.env.JWT_SECRET || 'default';

app.use(cors(corsOptions));//enable cors pe requests
app.options('*', cors(corsOptions));//enable cors pe preflight 

app.use(express.json({ limit: '50mb' }));//parse json also for large constellation data

//db and email null state
let db = null;
const dbConnectionMeta = {
    connected: false,
    error: 'Not initialized',
    configUsed: null,
};

let transporter = null;
const emailState = {
    configured: false,
    ready: false,
    error: 'Not initialized',
};

//tells the frontend if the server and database are running ok
app.get('/health', (req, res) => {
    const healthy = dbConnectionMeta.connected;
    res.status(healthy ? 200 : 503).json({
        status: healthy ? 'ok' : 'degraded',
        service: 'Node.js Express',
        dependencies: {
            database: {
                connected: dbConnectionMeta.connected,
                error: dbConnectionMeta.error,
                host: dbConnectionMeta.configUsed?.host || null,
                user: dbConnectionMeta.configUsed?.user || null,
                database: dbConnectionMeta.configUsed?.database || null,
            },
            email: {
                configured: emailState.configured,
                ready: emailState.ready,
                error: emailState.error,
            }
        }
    });
});

//user clicks the confirmation link in their email,this marks their account as confirmed
app.get('/confirm', (req, res) => {
    const { email } = req.query;

    if (!db) return res.status(503).send('Database unavailable.');

    db.getConnection((err, conn) => {
        if (err) return res.status(500).send('Database connection error');

        conn.query('UPDATE users SET confirmed = 1 WHERE email = ?', [email], (err, result) => {
            conn.release();
            if (err) return res.status(500).send('Error confirming email');
            if (result.affectedRows === 0) return res.status(400).send('Invalid confirmation link or user does not exist');
            return res.redirect('/confirmation-success.html');
        });
    });
});

// the signup page polls this every few seconds to see if the user confirmed their email yet
app.get('/is-confirmed', (req, res) => {
    const { email } = req.query;
    if (!email) return res.status(400).json({ confirmed: false, error: 'Email is required' });
    if (!db) return res.status(503).json({ confirmed: false, error: 'Database unavailable.' });

    db.getConnection((err, conn) => {
        if (err) return res.status(500).json({ confirmed: false, error: 'Database connection error' });

        conn.query('SELECT confirmed FROM users WHERE email = ?', [email], (err, results) => {
            conn.release();
            if (err) return res.status(500).json({ confirmed: false, error: 'Error checking status' });
            if (results.length === 0) return res.status(404).json({ confirmed: false, error: 'User not found' });
            return res.json({ confirmed: results[0].confirmed === 1 });
        });
    });
});

// serves the HTML files from the project folder
app.use(express.static(path.join(__dirname, '../../')));

// --- database setup ---

const dbBaseConfig = {
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
};

const dbPoolOptions = {
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
};

// builds a list of possible db credentials to try (handles root with/without password)
function buildDbCandidates() {
    const configured = {
        ...dbBaseConfig,
        password: typeof dbBaseConfig.password === 'string' ? dbBaseConfig.password : ''
    };
    const candidates = [configured];
    if (configured.user === 'root' && configured.password) {
        candidates.push({ ...configured, password: '' });
    }
    return candidates;
}

// connects to mysql, creates the database if it doesn't exist, and creates the tables
async function ensureDatabaseWithConfig(dbConfig) {
    return new Promise((resolve, reject) => {
        const adminConn = mysql.createConnection({
            host: dbConfig.host,
            user: dbConfig.user,
            password: dbConfig.password,
            multipleStatements: true
        });

        adminConn.connect(err => {
            if (err) { console.error('Cannot connect to MySQL:', err.message); return reject(err); }

            const dbName = mysql.escapeId(dbConfig.database);
            adminConn.query(`CREATE DATABASE IF NOT EXISTS ${dbName} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;`, (err) => {
                if (err) { console.error('Failed to create database:', err.message); adminConn.end(); return reject(err); }

                adminConn.changeUser({ database: dbConfig.database }, (err) => {
                    if (err) { console.error('Failed to switch to database:', err.message); adminConn.end(); return reject(err); }

                    // creates the users and constellations tables if they don't already exist
                    const createTablesSQL = `
                        CREATE TABLE IF NOT EXISTS users (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            username VARCHAR(255) NOT NULL,
                            email VARCHAR(255) NOT NULL UNIQUE,
                            password VARCHAR(255) NOT NULL,
                            confirmed TINYINT(1) NOT NULL DEFAULT 0,
                            profileImage VARCHAR(512) NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE TABLE IF NOT EXISTS constellations (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            user_id INT NOT NULL,
                            name VARCHAR(255) NOT NULL,
                            constellation_data LONGTEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                        );
                    `;

                    adminConn.query(createTablesSQL, (err) => {
                        if (err) { console.error('Failed to ensure tables:', err.message); adminConn.end(); return reject(err); }

                        // makes sure constellation_data is LONGTEXT (big enough for large constellations)
                        adminConn.query(`SHOW COLUMNS FROM constellations LIKE 'constellation_data'`, (checkErr, results) => {
                            if (checkErr) { adminConn.end(); console.warn('Could not check column:', checkErr.message); return resolve(); }

                            if (results.length === 0) {
                                adminConn.query(`ALTER TABLE constellations ADD COLUMN constellation_data LONGTEXT NOT NULL AFTER name;`, (migErr) => {
                                    adminConn.end();
                                    if (migErr) console.warn('Migration warning:', migErr.message);
                                    resolve();
                                });
                            } else {
                                adminConn.query(`ALTER TABLE constellations MODIFY COLUMN constellation_data LONGTEXT NOT NULL;`, (migErr) => {
                                    adminConn.end();
                                    if (migErr) console.warn('LONGTEXT upgrade failed:', migErr.message);
                                    resolve();
                                });
                            }
                        });
                    });
                });
            });
        });
    });
}

// tries each db config until one works, then creates a connection pool
async function initializeDatabase() {
    const candidates = buildDbCandidates();
    let lastError = null;

    for (const candidate of candidates) {
        try {
            await ensureDatabaseWithConfig(candidate);
            db = mysql.createPool({ ...candidate, ...dbPoolOptions });
            dbConnectionMeta.connected = true;
            dbConnectionMeta.error = null;
            dbConnectionMeta.configUsed = { host: candidate.host, user: candidate.user, database: candidate.database };
            console.log('Database pool ready.');
            return;
        } catch (err) {
            lastError = err;
            console.warn(`DB config failed for user "${candidate.user}" on "${candidate.host}".`);
        }
    }

    db = null;
    dbConnectionMeta.connected = false;
    dbConnectionMeta.error = lastError?.message || 'Unknown DB init error';
    dbConnectionMeta.configUsed = null;
    console.error('Database init failed. Check DB_* in .env.');
}

initializeDatabase();

// --- email setup ---

const emailUser = String(process.env.EMAIL || '').trim();
const emailPassword = String(process.env.EMAIL_PASSWORD || '').trim();

// sets up nodemailer with gmail credentials from .env so we can send confirmation emails
if (emailUser && emailPassword) {
    emailState.configured = true;
    transporter = nodemailer.createTransport({ service: 'gmail', auth: { user: emailUser, pass: emailPassword } });

    transporter.verify((error) => {
        if (error) {
            emailState.ready = false;
            emailState.error = error.message;
            console.error('Email transporter error:', error.message);
        } else {
            emailState.ready = true;
            emailState.error = null;
            console.log('Email ready.');
        }
    });
} else {
    emailState.configured = false;
    emailState.ready = false;
    emailState.error = 'Missing EMAIL or EMAIL_PASSWORD';
    console.warn('Email not configured. Signup will be blocked until EMAIL and EMAIL_PASSWORD are set in .env.');
}

// --- auth routes ---

// creates a new account — validates input, hashes the password, sends a confirmation email
app.post('/signup',
    [
        body('username').trim().isLength({ min: 3 }).withMessage('Username must be at least 3 characters'),
        body('email').isEmail().normalizeEmail().withMessage('Valid email required'),
        body('password').isLength({ min: 8 }).withMessage('Password must be at least 8 characters')
    ],
    async (req, res) => {
        const errors = validationResult(req);
        if (!errors.isEmpty()) return res.status(400).json({ errors: errors.array() });

        const { username, email, password } = req.body;

        if (!db) return res.status(503).json({ error: 'Database unavailable.' });

        try {
            db.getConnection((connErr, conn) => {
                if (connErr) return res.status(500).json({ error: 'Failed to connect to database' });

                conn.query('SELECT * FROM users WHERE email = ?', [email], async (err, result) => {
                    if (err) {
                        conn.release();
                        return res.status(500).json({ error: 'Failed to check user', detail: err.message });
                    }

                    if (result.length > 0) { conn.release(); return res.status(400).json({ error: 'Email already exists' }); }

                    // block signup entirely if email is not configured — no confirmation = no account
                    if (!emailState.ready) {
                        conn.release();
                        return res.status(503).json({ error: 'Email confirmation is required but the email service is not configured. Please contact the administrator.' });
                    }

                    // bcrypt hashes the password — we never store the real password
                    const hashedPassword = await bcrypt.hash(password, 12);

                    // account starts unconfirmed — user must click the link in their email
                    conn.query('INSERT INTO users (username, email, password, confirmed) VALUES (?, ?, ?, 0)',
                        [username, email, hashedPassword],
                        (err, result) => {
                            if (err) {
                                conn.release();
                                return res.status(500).json({ error: 'Failed to create user', detail: err.message });
                            }

                            const newUserId = result.insertId;
                            const appUrl = process.env.APP_URL || 'http://localhost:3002';
                            const confirmUrl = `${appUrl}/confirm?email=${encodeURIComponent(email)}`;
                            const mailOptions = {
                                from: process.env.EMAIL,
                                to: email,
                                subject: 'MindMap Email Confirmation',
                                html: `<h2>Welcome, ${username}!</h2><p>Please confirm your email by clicking the link below:</p><a href="${confirmUrl}">Confirm Email</a><p>Or copy and paste: ${confirmUrl}</p>`,
                            };

                            transporter.sendMail(mailOptions, (error, info) => {
                                if (error) {
                                    console.error('Error sending confirmation email:', error.message);

                                    // email failed — delete the account so the user can try again later
                                    conn.query('DELETE FROM users WHERE id = ?', [newUserId], (deleteErr) => {
                                        conn.release();
                                        return res.status(500).json({ error: 'Failed to send confirmation email. Please try again later.' });
                                    });
                                    return;
                                }

                                conn.release();
                                res.status(200).json({ message: 'Signup successful! Please check your email to confirm your account.', emailConfirmationRequired: true });
                            });
                        }
                    );
                });
            });
        } catch (error) {
            console.error('Unexpected error during signup:', error);
            res.status(500).json({ error: 'An unexpected error occurred' });
        }
    });

// checks email and password, returns a JWT token if correct
app.post('/login',
    [
        body('email').isEmail().withMessage('Valid email required'),
        body('password').isLength({ min: 8 }).withMessage('Password required')
    ],
    async (req, res) => {
        const errors = validationResult(req);
        if (!errors.isEmpty()) return res.status(400).json({ errors: errors.array() });

        const { email, password } = req.body;

        if (!db) return res.status(503).json({ error: 'Database unavailable.' });

        db.getConnection((err, conn) => {
            if (err) return res.status(500).json({ error: 'Server error' });

            // LOWER() on both sides so login works regardless of email casing
            conn.query('SELECT * FROM users WHERE LOWER(email) = LOWER(?)', [email], async (err, results) => {
                conn.release();

                if (err) return res.status(500).json({ error: 'Server error' });
                if (results.length === 0) return res.status(404).json({ error: 'User not found' });

                const user = results[0];

                if (user.confirmed !== 1) return res.status(400).json({ error: 'Please confirm your email before logging in.' });

                // bcrypt.compare checks the plain password against the stored hash
                const isPasswordCorrect = await bcrypt.compare(password, user.password);
                if (!isPasswordCorrect) return res.status(400).json({ error: 'Incorrect password' });

                // sign a token that expires in 1 hour — sent to the frontend and stored in localStorage
                const token = jwt.sign({ id: user.id, email: user.email }, JWT_SECRET, { expiresIn: '1h' });
                res.status(200).json({
                    message: 'Login successful',
                    token,
                    emailConfirmed: user.confirmed === 1,
                    user: { id: user.id, username: user.username, email: user.email }
                });
            });
        });
    });

// middleware that runs before protected routes — checks the JWT token from the Authorization header
function authenticateToken(req, res, next) {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    if (!token) return res.status(401).json({ error: 'Missing token' });

    jwt.verify(token, JWT_SECRET, (err, user) => {
        if (err) return res.status(403).json({ error: 'Invalid token' });
        req.user = user;
        next();
    });
}

// --- constellation routes ---

// saves a new constellation to the database
app.post('/save-constellation', authenticateToken, (req, res) => {
    const { name, constellationData } = req.body;
    const userId = req.user.id;

    if (!name || !constellationData) return res.status(400).json({ message: 'Name and constellation data are required.' });
    if (!db) return res.status(503).json({ message: 'Database unavailable.' });

    db.getConnection((err, conn) => {
        if (err) return res.status(500).json({ message: 'Error connecting to database.' });

        const dataToSave = typeof constellationData === 'string' ? constellationData : JSON.stringify(constellationData);

        conn.query('INSERT INTO constellations (user_id, name, constellation_data) VALUES (?, ?, ?)', [userId, name, dataToSave], (err, result) => {
            conn.release();
            if (err) { console.error('Error saving constellation:', err.message); return res.status(500).json({ message: 'Error saving constellation.', error: err.message }); }
            res.status(200).json({ message: 'Constellation saved.', constellationId: result.insertId });
        });
    });
});

// updates an existing constellation (name + data) when the user edits and saves again
app.put('/update-constellation/:id', authenticateToken, (req, res) => {
    const constellationId = req.params.id;
    const { name, constellationData } = req.body;

    if (!constellationId || !name || !constellationData) return res.status(400).json({ message: 'Constellation ID, name, and data are required.' });
    if (!db) return res.status(503).json({ message: 'Database unavailable.' });

    db.getConnection((err, conn) => {
        if (err) return res.status(500).json({ message: 'Error connecting to database.' });

        conn.query('UPDATE constellations SET name = ?, constellation_data = ? WHERE constellation_id = ?', [name, JSON.stringify(constellationData), constellationId], (err, result) => {
            conn.release();
            if (err) { console.error('Error updating constellation:', err.message); return res.status(500).json({ message: 'Error updating constellation.' }); }
            if (result.affectedRows === 0) return res.status(404).json({ message: 'Constellation not found.' });
            res.status(200).json({ message: 'Constellation updated successfully!' });
        });
    });
});

// deletes a constellation — checks that it belongs to the logged-in user before deleting
app.delete('/constellation/:id', authenticateToken, (req, res) => {
    const constellationId = req.params.id;
    const userId = req.user.id;

    if (!db) return res.status(503).json({ message: 'Database unavailable.' });

    db.getConnection((err, conn) => {
        if (err) return res.status(500).json({ message: 'Error connecting to database.' });

        conn.query('DELETE FROM constellations WHERE constellation_id = ? AND user_id = ?', [constellationId, userId], (err, result) => {
            conn.release();
            if (err) return res.status(500).json({ message: 'Error deleting constellation.' });
            if (result.affectedRows === 0) return res.status(404).json({ message: 'Constellation not found or not yours.' });
            res.status(200).json({ message: 'Constellation deleted.' });
        });
    });
});

// gets all constellations for a specific user — only works if you're requesting your own
app.get('/constellations/:userId', authenticateToken, (req, res) => {
    const userId = req.params.userId;

    if (req.user.id != userId) return res.status(403).json({ message: 'Unauthorized.' });

    db.getConnection((err, conn) => {
        if (err) return res.status(500).json({ error: 'Database connection error.' });

        conn.query('SELECT * FROM constellations WHERE user_id = ? ORDER BY created_at DESC', [userId], (err, results) => {
            conn.release();
            if (err) return res.status(500).json({ error: err.message });
            res.json(results);
        });
    });
});

// gets a single constellation by id — used when opening an existing constellation to edit
app.get('/constellation/:id', (req, res) => {
    const constellationId = req.params.id;

    if (!db) return res.status(503).json({ message: 'Database unavailable.' });

    db.getConnection((err, conn) => {
        if (err) return res.status(500).json({ message: 'Error connecting to database.' });

        conn.query(
            'SELECT constellation_id AS id, user_id, name, constellation_data, created_at FROM constellations WHERE constellation_id = ? LIMIT 1',
            [constellationId],
            (err, results) => {
                conn.release();
                if (err) return res.status(500).json({ message: 'Error fetching constellation.' });
                if (!results.length) return res.status(404).json({ message: 'Constellation not found.' });
                res.status(200).json(results[0]);
            }
        );
    });
});

// forwards the words to the Python Flask server which handles the NLP processing
app.post('/process-words', async (req, res) => {
    const { words } = req.body;
    if (!words) return res.status(400).json({ error: 'No words provided' });

    try {
        const flaskUrl = process.env.FLASK_URL || 'http://127.0.0.1:5000/process';
        const flaskResponse = await axios.post(flaskUrl, { words });
        res.json(flaskResponse.data);
    } catch (error) {
        console.error('Error communicating with Flask:', error);
        res.status(500).json({ error: 'Failed to process words. Ensure Flask is running.' });
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
