const express = require('express');
const path = require('path');
const nodemailer = require('nodemailer');
const dotenv = require('dotenv');
const bcrypt = require('bcryptjs');
const mysql = require('mysql2');
const cors = require('cors');
const axios = require('axios');
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
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Accept', 'Authorization'],
};

const app = express();
app.use((req, res, next) => {
    console.log(`[${new Date().toLocaleString()}] ${req.method} ${req.path}`);
    next();
});
const PORT = 3002;

app.use(helmet());

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100,
    standardHeaders: true,
    legacyHeaders: false,
});
app.use(limiter);

const JWT_SECRET = process.env.JWT_SECRET || 'please_change_this_to_a_strong_secret';

app.use(cors(corsOptions));
app.options('*', cors(corsOptions));

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

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
            if (err) { console.error('Cannot connect to MySQL:', err.message); return reject(err); }

            const dbName = mysql.escapeId(dbConfig.database);
            adminConn.query(`CREATE DATABASE IF NOT EXISTS ${dbName} DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;`, (err) => {
                if (err) { console.error('Failed to create database:', err.message); adminConn.end(); return reject(err); }

                adminConn.changeUser({ database: dbConfig.database }, (err) => {
                    if (err) { console.error('Failed to switch to database:', err.message); adminConn.end(); return reject(err); }

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
                            constellation_data LONGTEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                        );
                    `;

                    adminConn.query(createTablesSQL, (err) => {
                        if (err) { console.error('Failed to ensure tables:', err.message); adminConn.end(); return reject(err); }

                        // add status column if it is missing
                        adminConn.query(`SHOW COLUMNS FROM reports LIKE 'status'`, (colErr, colResults) => {
                            if (!colErr && colResults.length === 0) {
                                adminConn.query(
                                    `ALTER TABLE reports ADD COLUMN status ENUM('pending','reviewed','closed') NOT NULL DEFAULT 'pending'`,
                                    (migErr) => { if (migErr) console.warn('reports.status migration:', migErr.message); }
                                );
                            }
                        });

                        adminConn.query(`SHOW COLUMNS FROM constellations LIKE 'constellation_data'`, (checkErr, results) => {
                            if (checkErr) { adminConn.end(); console.warn('Could not check column:', checkErr.message); return resolve(); }

                            if (results.length === 0) {
                                adminConn.query(`ALTER TABLE constellations ADD COLUMN constellation_data LONGTEXT NOT NULL AFTER name;`, (migErr) => {
                                    adminConn.end();
                                    if (migErr) console.warn('Migration warning:', migErr.message);
                                    resolve();
                                });
                            } else {
                                // force LONGTEXT in case it was created as a smaller type
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

const emailUser = String(process.env.EMAIL || '').trim();
const emailPassword = String(process.env.EMAIL_PASSWORD || '').trim();

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
    console.warn('Email not configured. Signup runs without confirmation unless EMAIL_CONFIRMATION_REQUIRED=true.');
}

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
                        console.error('Database error during email check:', err.message);
                        return res.status(500).json({ error: 'Failed to check user', detail: err.message });
                    }

                    if (result.length > 0) { conn.release(); return res.status(400).json({ error: 'Email already exists' }); }

                    const hashedPassword = await bcrypt.hash(password, 12);

                    if (EMAIL_CONFIRMATION_REQUIRED && !emailState.ready) {
                        conn.release();
                        return res.status(503).json({ error: 'Email service unavailable. Set EMAIL_PASSWORD or disable EMAIL_CONFIRMATION_REQUIRED.' });
                    }

                    const initialConfirmed = emailState.ready ? 0 : 1;

                    conn.query('INSERT INTO users (username, email, password, confirmed) VALUES (?, ?, ?, ?)',
                        [username, email, hashedPassword, initialConfirmed],
                        (err, result) => {
                            if (err) {
                                conn.release();
                                console.error('Database error during user insert:', err.message);
                                return res.status(500).json({ error: 'Failed to create user', detail: err.message });
                            }

                            if (!emailState.ready) {
                                conn.release();
                                return res.status(200).json({ message: 'Signup successful! Email confirmation is disabled in local mode.', emailConfirmationRequired: false });
                            }

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
                                    if (EMAIL_CONFIRMATION_REQUIRED) { conn.release(); return res.status(500).json({ error: 'Error sending confirmation email', detail: error.message }); }

                                    conn.query('UPDATE users SET confirmed = 1 WHERE id = ?', [result.insertId], (updateErr) => {
                                        conn.release();
                                        if (updateErr) return res.status(200).json({ message: 'Signup successful, but email failed. Contact support.', emailConfirmationRequired: true });
                                        return res.status(200).json({ message: 'Signup successful! Email failed, account auto-confirmed for local dev.', emailConfirmationRequired: false });
                                    });
                                    return;
                                }

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
        if (!errors.isEmpty()) return res.status(400).json({ errors: errors.array() });

        const { email, password } = req.body;

        if (!db) return res.status(503).json({ error: 'Database unavailable.' });

        db.getConnection((err, conn) => {
            if (err) return res.status(500).json({ error: 'Server error' });

            conn.query('SELECT * FROM users WHERE email = ?', [email], async (err, results) => {
                conn.release();

                if (err) return res.status(500).json({ error: 'Server error' });
                if (results.length === 0) return res.status(404).json({ error: 'User not found' });

                const user = results[0];

                if (user.confirmed !== 1) return res.status(400).json({ error: 'Please confirm your email before logging in.' });

                const isPasswordCorrect = await bcrypt.compare(password, user.password);
                if (!isPasswordCorrect) return res.status(400).json({ error: 'Incorrect password' });

                // email in token so admin check works server-side
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

// auth middleware
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

    if (!report) return res.status(400).json({ message: 'Report content is required.' });
    if (!db) return res.status(503).json({ message: 'Database unavailable.' });

    db.getConnection((err, conn) => {
        if (err) return res.status(500).json({ message: 'Error submitting the report.' });

        conn.query('INSERT INTO reports (user_id, report_content) VALUES (?, ?)', [id, report], (err) => {
            conn.release();
            if (err) { console.error('Error saving report:', err.message); return res.status(500).json({ message: 'Error submitting the report.' }); }
            res.status(200).json({ message: 'Report submitted successfully.' });
        });
    });
});

// admin
const ADMIN_EMAILS = (process.env.ADMIN_EMAIL || '')
    .split(',').map(e => e.trim().toLowerCase()).filter(Boolean);

function requireAdmin(req, res, next) {
    if (!ADMIN_EMAILS.includes((req.user.email || '').toLowerCase())) {
        return res.status(403).json({ message: 'Admin access required.' });
    }
    next();
}

// admin ping
app.get('/admin/check', authenticateToken, requireAdmin, (req, res) => {
    res.json({ admin: true });
});

// get all reports with user info
app.get('/admin/reports', authenticateToken, requireAdmin, (req, res) => {
    if (!db) return res.status(503).json({ message: 'Database unavailable.' });
    db.getConnection((err, conn) => {
        if (err) return res.status(500).json({ message: 'DB connection error.' });

        // detect actual column names (schema may vary)
        conn.query(`SHOW COLUMNS FROM reports`, (colErr, columns) => {
            if (colErr) {
                conn.release();
                console.error('Error reading reports schema:', colErr.message);
                return res.status(500).json({ message: 'Error reading reports schema.', detail: colErr.message });
            }

            const fields = columns.map(c => c.Field);
            const idField      = fields.includes('id')             ? 'reports.id'             : fields.find(f => f.endsWith('_id') && f !== 'user_id') || fields[0];
            const contentField = fields.includes('report_content') ? 'reports.report_content' : fields.includes('content') ? 'reports.content' : 'reports.' + fields[1];
            const statusField  = fields.includes('status')         ? 'reports.status'         : `'pending' AS status`;
            const createdField = fields.includes('created_at')     ? 'reports.created_at'     : `NULL AS created_at`;

            const query = `
                SELECT ${idField} AS id,
                       ${contentField} AS report_content,
                       ${statusField},
                       ${createdField},
                       reports.user_id,
                       users.username,
                       users.email
                FROM reports
                JOIN users ON reports.user_id = users.id
                ORDER BY reports.created_at DESC`;

            conn.query(query, (err, rows) => {
                conn.release();
                if (err) {
                    console.error('Error fetching admin reports:', err.message);
                    return res.status(500).json({ message: 'Error fetching reports.', detail: err.message });
                }
                res.json(rows);
            });
        });
    });
});

// update report status
app.patch('/admin/report/:id/status', authenticateToken, requireAdmin, (req, res) => {
    const { status } = req.body;
    if (!['pending', 'reviewed', 'closed'].includes(status)) return res.status(400).json({ message: 'Invalid status.' });
    if (!db) return res.status(503).json({ message: 'Database unavailable.' });

    db.getConnection((err, conn) => {
        if (err) return res.status(500).json({ message: 'DB connection error.' });

        conn.query(`SHOW COLUMNS FROM reports`, (colErr, columns) => {
            if (colErr) { conn.release(); return res.status(500).json({ message: 'Error reading schema.', detail: colErr.message }); }

            const fields  = columns.map(c => c.Field);
            const idField = fields.includes('id') ? 'id' : fields.find(f => f.endsWith('_id') && f !== 'user_id') || fields[0];
            const hasStatus = fields.includes('status');

            const doUpdate = () => {
                conn.query(`UPDATE reports SET status = ? WHERE ${idField} = ?`, [status, req.params.id], (err, result) => {
                    conn.release();
                    if (err) { console.error('Error updating report status:', err.message); return res.status(500).json({ message: 'Error updating status.', detail: err.message }); }
                    if (result.affectedRows === 0) return res.status(404).json({ message: 'Report not found.' });
                    res.json({ message: 'Status updated.' });
                });
            };

            if (!hasStatus) {
                conn.query(`ALTER TABLE reports ADD COLUMN status ENUM('pending','reviewed','closed') NOT NULL DEFAULT 'pending'`, (migErr) => {
                    if (migErr) { conn.release(); return res.status(500).json({ message: 'Could not add status column.', detail: migErr.message }); }
                    doUpdate();
                });
            } else {
                doUpdate();
            }
        });
    });
});

app.post('/save-constellation', authenticateToken, (req, res) => {
    const { name, constellationData } = req.body;
    const userId = req.user.id;

    if (!name || !constellationData) return res.status(400).json({ message: 'Name and constellation data are required.' });
    if (!db) return res.status(503).json({ message: 'Database unavailable.' });

    db.getConnection((err, conn) => {
        if (err) { console.error('Database connection error:', err.message); return res.status(500).json({ message: 'Error connecting to database.' }); }

        const dataToSave = typeof constellationData === 'string' ? constellationData : JSON.stringify(constellationData);

        conn.query('INSERT INTO constellations (user_id, name, constellation_data) VALUES (?, ?, ?)', [userId, name, dataToSave], (err, result) => {
            conn.release();
            if (err) { console.error('Error saving constellation:', err.message); return res.status(500).json({ message: 'Error saving constellation.', error: err.message }); }
            res.status(200).json({ message: 'Constellation saved.', constellationId: result.insertId });
        });
    });
});

app.put('/update-constellation/:id', authenticateToken, (req, res) => {
    const constellationId = req.params.id;
    const { name, constellationData } = req.body;

    if (!constellationId || !name || !constellationData) return res.status(400).json({ message: 'Constellation ID, name, and data are required.' });
    if (!db) return res.status(503).json({ message: 'Database unavailable.' });

    db.getConnection((err, conn) => {
        if (err) { console.error('Database connection error:', err); return res.status(500).json({ message: 'Error connecting to database.' }); }

        conn.query('SHOW COLUMNS FROM constellations', (err, columns) => {
            if (err) { conn.release(); return res.status(500).json({ message: 'Error updating constellation.' }); }

            const columnNames = columns.map(c => c.Field);
            const idField = columnNames.includes('constellation_id') ? 'constellation_id' : 'id';

            conn.query(`UPDATE constellations SET name = ?, constellation_data = ? WHERE ${idField} = ?`, [name, JSON.stringify(constellationData), constellationId], (err, result) => {
                conn.release();
                if (err) { console.error('Error updating constellation:', err.message); return res.status(500).json({ message: 'Error updating constellation.' }); }
                if (result.affectedRows === 0) return res.status(404).json({ message: 'Constellation not found.' });
                res.status(200).json({ message: 'Constellation updated successfully!' });
            });
        });
    });
});

app.delete('/constellation/:id', authenticateToken, (req, res) => {
    const constellationId = req.params.id;
    const userId = req.user.id;

    if (!db) return res.status(503).json({ message: 'Database unavailable.' });

    db.getConnection((err, conn) => {
        if (err) return res.status(500).json({ message: 'Error connecting to database.' });

        conn.query('SHOW COLUMNS FROM constellations', (err, columns) => {
            if (err) { conn.release(); return res.status(500).json({ message: 'Error deleting constellation.' }); }

            const columnNames = columns.map(c => c.Field);
            const idField = columnNames.includes('constellation_id') ? 'constellation_id' : 'id';

            conn.query(`DELETE FROM constellations WHERE ${idField} = ? AND user_id = ?`, [constellationId, userId], (err, result) => {
                conn.release();
                if (err) return res.status(500).json({ message: 'Error deleting constellation.' });
                if (result.affectedRows === 0) return res.status(404).json({ message: 'Constellation not found or not yours.' });
                res.status(200).json({ message: 'Constellation deleted.' });
            });
        });
    });
});

app.get('/constellations/:userId', authenticateToken, (req, res) => {
    const userId = req.params.userId;

    if (req.user.id != userId) return res.status(403).json({ message: 'Unauthorized.' });

    db.getConnection((err, conn) => {
        if (err) { console.error('Database connection error:', err); return res.status(500).json({ error: 'Database connection error.' }); }

        conn.query('SELECT * FROM constellations WHERE user_id = ? ORDER BY created_at DESC', [userId], (err, results) => {
            conn.release();
            if (err) { console.error('Query error:', err.message); return res.status(500).json({ error: err.message }); }
            res.json(results);
        });
    });
});

app.get('/constellation/:id', (req, res) => {
    const constellationId = req.params.id;

    if (!db) return res.status(503).json({ message: 'Database unavailable.' });

    db.getConnection((err, conn) => {
        if (err) return res.status(500).json({ message: 'Error connecting to database.' });

        conn.query('SHOW COLUMNS FROM constellations', (err, columns) => {
            if (err) { conn.release(); return res.status(500).json({ message: 'Error fetching constellation.' }); }

            const columnNames = columns.map(c => c.Field);
            const selectFields = [];
            if (columnNames.includes('constellation_id')) selectFields.push('constellation_id as id');
            else if (columnNames.includes('id')) selectFields.push('id');
            if (columnNames.includes('user_id')) selectFields.push('user_id');
            if (columnNames.includes('name')) selectFields.push('name');
            if (columnNames.includes('constellation_data')) selectFields.push('constellation_data');
            if (columnNames.includes('created_at')) selectFields.push('created_at');

            const idCol = columnNames.includes('constellation_id') ? 'constellation_id' : 'id';
            const query = `SELECT ${selectFields.join(', ')} FROM constellations WHERE ${idCol} = ? LIMIT 1`;

            conn.query(query, [constellationId], (err, results) => {
                conn.release();
                if (err) { console.error('Error fetching constellation:', err.message); return res.status(500).json({ message: 'Error fetching constellation.' }); }
                if (!results.length) return res.status(404).json({ message: 'Constellation not found.' });
                res.status(200).json(results[0]);
            });
        });
    });
});

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
