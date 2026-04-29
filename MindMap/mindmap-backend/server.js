const express = require('express');
const path = require('path');
const nodemailer = require('nodemailer');
const dotenv = require('dotenv');
const bcrypt = require('bcryptjs');
const mysql = require('mysql2');
const cors = require('cors');
const axios = require('axios');

// security-related packages
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const jwt = require('jsonwebtoken');
const { body, validationResult } = require('express-validator');

dotenv.config();

const EMAIL_CONFIRMATION_REQUIRED = String(process.env.EMAIL_CONFIRMATION_REQUIRED || '').toLowerCase() === 'true';

const corsOptions = {
    origin: (origin, callback) => {
        return callback(null, true);
    },
    methods: ['GET', 'POST', 'PUT', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Accept'],
};


const app = express();
const PORT = 3002;

// basic security headers
app.use(helmet());

// simple rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP
    standardHeaders: true,
    legacyHeaders: false,
});
app.use(limiter);

// JWT secret must be set in environment for production
const JWT_SECRET = process.env.JWT_SECRET || 'please_change_this_to_a_strong_secret';

app.use(cors(corsOptions));
app.options('*', cors(corsOptions));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

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

app.get('/confirm', (req, res) => {
    const { email } = req.query;
    console.log('Email confirmation received for:', email);

    if (!db) {
        return res.status(503).send('Database unavailable. Please check MySQL connection.');
    }

    db.getConnection((err, conn) => {
        if (err) {
            console.error('Database connection error:', err);
            return res.status(500).send('Database connection error');
        }

        conn.query(
            'UPDATE users SET confirmed = 1 WHERE email = ?',
            [email],
            (err, result) => {
                conn.release();
                
                if (err) {
                    console.error('Error confirming email:', err);
                    return res.status(500).send('Error confirming email');
                }

                if (result.affectedRows === 0) {
                    console.log('No user found with the provided email');
                    return res.status(400).send('Invalid confirmation link or user does not exist');
                }

                console.log('Email confirmed successfully');
                return res.redirect('/confirmation-success.html');
            }
        );
    });
});

app.get('/is-confirmed', (req, res) => {
    const { email } = req.query;
    if (!email) return res.status(400).json({ confirmed: false, error: 'Email is required' });

    if (!db) {
        return res.status(503).json({ confirmed: false, error: 'Database unavailable. Please check MySQL connection.' });
    }

    db.getConnection((err, conn) => {
        if (err) {
            console.error('Database connection error:', err);
            return res.status(500).json({ confirmed: false, error: 'Database connection error' });
        }

        conn.query('SELECT confirmed FROM users WHERE email = ?', [email], (err, results) => {
            conn.release();
            if (err) {
                console.error('Error checking confirmation status:', err);
                return res.status(500).json({ confirmed: false, error: 'Error checking status' });
            }

            if (results.length === 0) {
                return res.status(404).json({ confirmed: false, error: 'User not found' });
            }

            const confirmed = results[0].confirmed === 1;
            return res.json({ confirmed });
        });
    });
});

app.use(express.static(path.join(__dirname, '../../')));

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

async function ensureDatabaseWithConfig(dbConfig) {
    return new Promise((resolve, reject) => {
        const adminConn = mysql.createConnection({
            host: dbConfig.host,
            user: dbConfig.user,
            password: dbConfig.password,
            multipleStatements: true
        });

        adminConn.connect(err => {
            if (err) {
                console.error('❌ Cannot connect to MySQL server:', err.message);
                return reject(err);
            }

            const dbName = mysql.escapeId(dbConfig.database);
            const createDbSQL = `CREATE DATABASE IF NOT EXISTS ${dbName} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;`;
            adminConn.query(createDbSQL, (err) => {
                if (err) {
                    console.error('❌ Failed to create database if missing:', err.message);
                    adminConn.end();
                    return reject(err);
                }

                adminConn.changeUser({ database: dbConfig.database }, (err) => {
                    if (err) {
                        console.error('❌ Failed to switch to database:', err.message);
                        adminConn.end();
                        return reject(err);
                    }

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
                        CREATE TABLE IF NOT EXISTS reports (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            user_id INT NOT NULL,
                            report_content TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                        );
                        CREATE TABLE IF NOT EXISTS constellations (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            user_id INT NOT NULL,
                            name VARCHAR(255) NOT NULL,
                            constellation_data TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                        );
                    `;

                    adminConn.query(createTablesSQL, (err) => {
                        if (err) {
                            console.error('❌ Failed to ensure tables exist:', err.message);
                            adminConn.end();
                            return reject(err);
                        }
                        
                        adminConn.query(`SHOW COLUMNS FROM constellations LIKE 'constellation_data'`, (checkErr, results) => {
                            if (checkErr) {
                                adminConn.end();
                                console.warn('⚠️ Could not check column:', checkErr.message);
                                return resolve();
                            }
                            
                            if (results.length === 0) {
                                adminConn.query(`ALTER TABLE constellations ADD COLUMN constellation_data TEXT NOT NULL AFTER name;`, (migErr) => {
                                    adminConn.end();
                                    if (migErr) {
                                        console.warn('⚠️ Migration warning:', migErr.message);
                                    } else {
                                        console.log('✅ Added constellation_data column.');
                                    }
                                    console.log('✅ Database and tables are ready.');
                                    resolve();
                                });
                            } else {
                                adminConn.end();
                                console.log('✅ Database and tables are ready.');
                                resolve();
                            }
                        });
                    });
                });
            });
        });
    });
}

async function initializeDatabase() {
    const candidates = buildDbCandidates();
    let lastError = null;

    for (const candidate of candidates) {
        try {
            await ensureDatabaseWithConfig(candidate);
            db = mysql.createPool({ ...candidate, ...dbPoolOptions });
            dbConnectionMeta.connected = true;
            dbConnectionMeta.error = null;
            dbConnectionMeta.configUsed = {
                host: candidate.host,
                user: candidate.user,
                database: candidate.database,
            };
            console.log('✅ Database pool initialized successfully.');
            return;
        } catch (err) {
            lastError = err;
            console.warn(`⚠️ Database config attempt failed for user "${candidate.user}" on host "${candidate.host}".`);
        }
    }

    db = null;
    dbConnectionMeta.connected = false;
    dbConnectionMeta.error = lastError?.message || 'Unknown DB init error';
    dbConnectionMeta.configUsed = null;
    console.error('❌ Database initialization failed. Check DB_* credentials in .env.');
}

initializeDatabase();

const emailUser = String(process.env.EMAIL || '').trim();
const emailPassword = String(process.env.EMAIL_PASSWORD || '').trim();

if (emailUser && emailPassword) {
    emailState.configured = true;
    transporter = nodemailer.createTransport({
        service: 'gmail',
        auth: {
            user: emailUser,
            pass: emailPassword,
        },
    });

    transporter.verify((error) => {
        if (error) {
            emailState.ready = false;
            emailState.error = error.message;
            console.error('Error configuring email transporter:', error.message);
        } else {
            emailState.ready = true;
            emailState.error = null;
            console.log('Email transporter is ready to send messages');
        }
    });
} else {
    emailState.configured = false;
    emailState.ready = false;
    emailState.error = 'Missing EMAIL or EMAIL_PASSWORD';
    console.warn('⚠️ Email is not configured. Signup will continue without email confirmation unless EMAIL_CONFIRMATION_REQUIRED=true.');
}

app.post('/signup',
    // validation and sanitization
    [
        body('username').trim().isLength({ min: 3 }).withMessage('Username must be at least 3 characters'),
        body('email').isEmail().normalizeEmail().withMessage('Valid email required'),
        body('password').isLength({ min: 8 }).withMessage('Password must be at least 8 characters')
    ],
    async (req, res) => {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({ errors: errors.array() });
        }

        const { username, email, password } = req.body;
        console.log('Signup request received:', { username, email });

    if (!db) {
        return res.status(503).json({ error: 'Database unavailable. Please check MySQL connection.' });
    }

    if (!db) {
        return res.status(503).json({ error: 'Database unavailable. Please check MySQL connection.' });
    }

    try {
        console.log('Checking if user already exists in the database...');
        db.getConnection((connErr, conn) => {
            if (connErr) {
                console.error('Database connection error:', connErr);
                return res.status(500).json({ error: 'Failed to connect to database' });
            }

            conn.query('SELECT * FROM users WHERE email = ?', [email], async (err, result) => {
                if (err) {
                    conn.release();
                    console.error('❌ Database error during email check:', err.message);
                    console.error('🛠️ Full error object:', err);
                    return res.status(500).json({ error: 'Failed to check user', detail: err.message });
                }


                if (result.length > 0) {
                    conn.release();
                    console.log('Email already exists');
                    return res.status(400).json({ error: 'Email already exists' });
                }

                console.log('Hashing password...');
                const hashedPassword = await bcrypt.hash(password, 12); // stronger cost factor

                if (EMAIL_CONFIRMATION_REQUIRED && !emailState.ready) {
                    conn.release();
                    return res.status(503).json({
                        error: 'Email service unavailable. Set a valid EMAIL_PASSWORD (Gmail app password) or disable EMAIL_CONFIRMATION_REQUIRED.'
                    });
                }

                const initialConfirmed = emailState.ready ? 0 : 1;

                console.log('Inserting new user into the database...');
                conn.query(
                    'INSERT INTO users (username, email, password, confirmed) VALUES (?, ?, ?, ?)',
                    [username, email, hashedPassword, initialConfirmed],
                    (err, result) => {
                        if (err) {
                            conn.release();
                            console.error('❌ Database error during user insert:', err.message);
                            console.error('🛠️ Full error object:', err);
                            return res.status(500).json({ error: 'Failed to create user', detail: err.message });
                        }

                        if (!emailState.ready) {
                            conn.release();
                            return res.status(200).json({
                                message: 'Signup successful! Email confirmation is disabled in local mode.',
                                emailConfirmationRequired: false
                            });
                        }

                        console.log('Sending confirmation email...');
                        const appUrl = process.env.APP_URL || 'http://localhost:3002';
                        const confirmUrl = `${appUrl}/confirm?email=${encodeURIComponent(email)}`;
                        const mailOptions = {
                            from: process.env.EMAIL,
                            to: email,
                            subject: 'MindMap Email Confirmation',
                            html: `
                                <h2>Welcome, ${username}!</h2>
                                <p>Please confirm your email by clicking the link below:</p>
                                <a href="${confirmUrl}">Confirm Email</a>
                                <p>Or copy and paste: ${confirmUrl}</p>
                            `,
                        };

                        transporter.sendMail(mailOptions, (error, info) => {
                            if (error) {
                                console.error('Error sending confirmation email:', error.message);
                                if (EMAIL_CONFIRMATION_REQUIRED) {
                                    conn.release();
                                    return res.status(500).json({ error: 'Error sending confirmation email', detail: error.message });
                                }

                                conn.query('UPDATE users SET confirmed = 1 WHERE id = ?', [result.insertId], (updateErr) => {
                                    conn.release();
                                    if (updateErr) {
                                        return res.status(200).json({
                                            message: 'Signup successful, but confirmation email failed. Please contact support to confirm your account.',
                                            emailConfirmationRequired: true
                                        });
                                    }

                                    return res.status(200).json({
                                        message: 'Signup successful! Email delivery failed, so your account was auto-confirmed for local development.',
                                        emailConfirmationRequired: false
                                    });
                                });
                                return;
                            }

                            console.log('Confirmation email sent:', info.response);
                            conn.release();
                            res.status(200).json({ message: 'Signup successful! Please check your email.', emailConfirmationRequired: true });
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

app.post('/login',
    [
        body('email').isEmail().normalizeEmail().withMessage('Valid email required'),
        body('password').isLength({ min: 8 }).withMessage('Password required')
    ],
    async (req, res) => {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({ errors: errors.array() });
        }

        console.log("Login route hit");
        const { email, password } = req.body;
            console.log('Login request received:', { email });

            if (!db) {
                return res.status(503).json({ error: 'Database unavailable. Please check MySQL connection.' });
            }

    db.getConnection((err, conn) => {
        if (err) {
            console.error('Database connection error:', err);
            return res.status(500).json({ error: 'Server error' });
        }

        conn.query('SELECT * FROM users WHERE email = ?', [email], async (err, results) => {
            conn.release();
            
            if (err) {
                console.error('Database error during login:', err);
                return res.status(500).json({ error: 'Server error' });
            }

            if (results.length === 0) {
                console.log('User not found for email:', email);
                return res.status(404).json({ error: 'User not found' });
            }

            const user = results[0];
            console.log('User found in database:', { username: user.username, email: user.email, confirmed: user.confirmed });

            console.log('Checking if user is confirmed...');
            if (user.confirmed !== 1) {
                console.log('User has not confirmed email:', email);
                return res.status(400).json({ error: 'Please confirm your email before logging in.' });
            }

            console.log('Verifying password...');
            const isPasswordCorrect = await bcrypt.compare(password, user.password);
            if (!isPasswordCorrect) {
                console.log('Incorrect password for user:', email);
                return res.status(400).json({ error: 'Incorrect password' });
            }

            console.log('Login successful for user:', email);
            // issue JWT token
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

// middleware to guard endpoints
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

app.post('/submit-report', authenticateToken, (req, res) => {
    const { report } = req.body;
    const id = req.user.id;

    console.log('Received report from user', id, { report });

    if (!report) {
        return res.status(400).json({ message: 'Report content is required.' });
    }

    if (!db) {
        return res.status(503).json({ message: 'Database unavailable. Please check MySQL connection.' });
    }

    db.getConnection((err, conn) => {
        if (err) {
            console.error('Database connection error:', err);
            return res.status(500).json({ message: 'Error submitting the report.' });
        }

        const query = 'INSERT INTO reports (user_id, report_content) VALUES (?, ?)';
        conn.query(query, [id, report], (err, result) => {
            conn.release();
            if (err) {
                console.error('Error saving the report: ' + err.message);
                return res.status(500).json({ message: 'Error submitting the report.' });
            }

            console.log('Report saved successfully:', result);
            res.status(200).json({ message: 'Report submitted successfully.' });
        });
    });
});

app.post('/save-constellation', authenticateToken, (req, res) => {
    const { name, constellationData } = req.body;
    const userId = req.user.id;
    
    console.log('Received save constellation request from user', userId, { name });
    
    if (!name || !constellationData) {
        return res.status(400).json({ message: 'Name and constellation data are required.' });
    }
    
    if (!db) {
        return res.status(503).json({ message: 'Database unavailable. Please check MySQL connection.' });
    }
    
    db.getConnection((err, conn) => {
        if (err) {
            console.error('Database connection error:', err);
            return res.status(500).json({ message: 'Error connecting to database.' });
        }
        
        const query = 'INSERT INTO constellations (user_id, name, constellation_data) VALUES (?, ?, ?)';
        conn.query(query, [userId, name, JSON.stringify(constellationData)], (err, result) => {
            conn.release();
            if (err) {
                console.error('Error saving constellation:', err.message);
                return res.status(500).json({ message: 'Error saving constellation.' });
            }
            
            console.log('Constellation saved successfully:', result);
            res.status(200).json({ message: 'Constellation saved successfully!', constellationId: result.insertId });
        });
    });
});

app.put('/update-constellation/:id', authenticateToken, (req, res) => {
    const constellationId = req.params.id;
    const { name, constellationData } = req.body;

    if (!constellationId || !name || !constellationData) {
        return res.status(400).json({ message: 'Constellation ID, name, and constellation data are required.' });
    }

    if (!db) {
        return res.status(503).json({ message: 'Database unavailable. Please check MySQL connection.' });
    }

    db.getConnection((err, conn) => {
        if (err) {
            console.error('Database connection error:', err);
            return res.status(500).json({ message: 'Error connecting to database.' });
        }

        conn.query('SHOW COLUMNS FROM constellations', (err, columns) => {
            if (err) {
                conn.release();
                console.error('Error checking columns:', err.message);
                return res.status(500).json({ message: 'Error updating constellation.' });
            }

            const columnNames = columns.map(c => c.Field);
            const idField = columnNames.includes('constellation_id') ? 'constellation_id' : 'id';

            const query = `UPDATE constellations SET name = ?, constellation_data = ? WHERE ${idField} = ?`;
            conn.query(query, [name, JSON.stringify(constellationData), constellationId], (err, result) => {
                conn.release();
                if (err) {
                    console.error('Error updating constellation:', err.message);
                    return res.status(500).json({ message: 'Error updating constellation.' });
                }

                if (result.affectedRows === 0) {
                    return res.status(404).json({ message: 'Constellation not found.' });
                }

                res.status(200).json({ message: 'Constellation updated successfully!' });
            });
        });
    });
});

app.get('/get-constellations/:userId', (req, res) => {
    const userId = req.params.userId;
    
    if (!db) {
        return res.status(503).json({ message: 'Database unavailable. Please check MySQL connection.' });
    }
    
    db.getConnection((err, conn) => {
        if (err) {
            console.error('Database connection error:', err);
            return res.status(500).json({ message: 'Error connecting to database.' });
        }
        
        conn.query('SHOW COLUMNS FROM constellations', (err, columns) => {
            if (err) {
                console.error('Error checking columns:', err.message);
                conn.release();
                return res.status(500).json({ message: 'Error checking table structure.' });
            }
            
            const columnNames = columns.map(col => col.Field);
            console.log('Available columns in constellations:', columnNames);
            
            const selectFields = [];
            if (columnNames.includes('constellation_id')) selectFields.push('constellation_id as id');
            else if (columnNames.includes('id')) selectFields.push('id');
            
            if (columnNames.includes('name')) selectFields.push('name');
            if (columnNames.includes('constellation_data')) selectFields.push('constellation_data');
            if (columnNames.includes('created_at')) selectFields.push('created_at');
            
            if (selectFields.length === 0) {
                selectFields.push('*');
            }
            
            const query = `SELECT ${selectFields.join(', ')} FROM constellations WHERE user_id = ? ORDER BY created_at DESC`;
            
            conn.query(query, [userId], (err, results) => {
                conn.release();
                if (err) {
                    console.error('Error fetching constellations:', err.message);
                    return res.status(500).json({ message: 'Error fetching constellations.' });
                }
                
                res.status(200).json(results);
            });
        });
    });
});

app.get('/constellation/:id', (req, res) => {
    const constellationId = req.params.id;

    if (!db) {
        return res.status(503).json({ message: 'Database unavailable. Please check MySQL connection.' });
    }

    db.getConnection((err, conn) => {
        if (err) {
            console.error('Database connection error:', err);
            return res.status(500).json({ message: 'Error connecting to database.' });
        }

        conn.query('SHOW COLUMNS FROM constellations', (err, columns) => {
            if (err) {
                conn.release();
                console.error('Error checking columns:', err.message);
                return res.status(500).json({ message: 'Error fetching constellation.' });
            }

            const columnNames = columns.map(c => c.Field);
            const selectFields = [];
            if (columnNames.includes('constellation_id')) selectFields.push('constellation_id as id');
            else if (columnNames.includes('id')) selectFields.push('id');

            if (columnNames.includes('user_id')) selectFields.push('user_id');
            if (columnNames.includes('name')) selectFields.push('name');
            if (columnNames.includes('constellation_data')) selectFields.push('constellation_data');
            if (columnNames.includes('created_at')) selectFields.push('created_at');

            const query = `SELECT ${selectFields.join(', ')} FROM constellations WHERE ${columnNames.includes('constellation_id') ? 'constellation_id' : 'id'} = ? LIMIT 1`;

            conn.query(query, [constellationId], (err, results) => {
                conn.release();
                if (err) {
                    console.error('Error fetching constellation:', err.message);
                    return res.status(500).json({ message: 'Error fetching constellation.' });
                }

                if (!results.length) {
                    return res.status(404).json({ message: 'Constellation not found.' });
                }

                res.status(200).json(results[0]);
            });
        });
    });
});


app.post('/process-words', async (req, res) => {
    console.log('Received POST request to /process-words');
    console.log('Request Body:', req.body);

    const { words } = req.body;
    if (!words) {
        return res.status(400).json({ error: 'No words provided' });
    }

    try {
        const flaskUrl = process.env.FLASK_URL || 'http://127.0.0.1:5000/process';
        const flaskResponse = await axios.post(flaskUrl, { words });

        console.log("Flask response:", flaskResponse.data);
        res.json(flaskResponse.data);
    } catch (error) {
        console.error("Error communicating with Flask API:", error);
        res.status(500).json({ error: "Failed to process words. Ensure Flask is running." });
    }
});


app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running at http://localhost:${PORT}`);
});