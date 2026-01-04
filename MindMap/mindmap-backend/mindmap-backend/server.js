const express = require('express');
const path = require('path');
const nodemailer = require('nodemailer');
const dotenv = require('dotenv');
const bcrypt = require('bcryptjs');
const mysql = require('mysql2');
const cors = require('cors');
const axios = require('axios');

dotenv.config();

const allowedOrigins = (process.env.FRONTEND_ORIGIN || '').split(',').map(o => o.trim()).filter(Boolean);
const corsDefaults = ['http://localhost:3000', 'http://localhost:3002'];
const corsOptions = {
    origin: (origin, callback) => {
        if (!origin || allowedOrigins.includes(origin) || corsDefaults.includes(origin)) {
            return callback(null, true);
        }
        return callback(new Error('Not allowed by CORS'));
    },
    methods: ['GET', 'POST', 'PUT', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Accept'],
};


const app = express();
const PORT = 3002;

app.use(cors(corsOptions));
app.options('*', cors(corsOptions));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/confirm', (req, res) => {
    const { email } = req.query;
    console.log('Email confirmation received for:', email);

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

const dbConfig = {
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
};

async function ensureDatabase() {
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

let db;
ensureDatabase()
    .then(() => {
        db = mysql.createPool(dbConfig);
    })
    .catch(err => {
        console.error('❌ Database initialization failed. Check credentials / MySQL server.', err);
    });

const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: process.env.EMAIL,
        pass: process.env.EMAIL_PASSWORD,
    },
});

transporter.verify((error, success) => {
    if (error) {
        console.error('Error configuring email transporter:', error);
    } else {
        console.log('Email transporter is ready to send messages');
    }
});

app.post('/signup', async (req, res) => {
    const { username, email, password } = req.body;
    console.log('Signup request received:', { username, email });

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
                const hashedPassword = await bcrypt.hash(password, 10);

                console.log('Inserting new user into the database...');
                conn.query(
                    'INSERT INTO users (username, email, password, confirmed) VALUES (?, ?, ?, ?)',
                    [username, email, hashedPassword, 0],
                    (err, result) => {
                        if (err) {
                            conn.release();
                            console.error('❌ Database error during user insert:', err.message);
                            console.error('🛠️ Full error object:', err);
                            return res.status(500).json({ error: 'Failed to create user', detail: err.message });
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
                            conn.release();
                            if (error) {
                                console.error('Error sending confirmation email:', error);
                                return res.status(500).json({ error: 'Error sending confirmation email', detail: error.message });
                            }

                            console.log('Confirmation email sent:', info.response);
                            res.status(200).json({ message: 'Signup successful! Please check your email.' });
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

app.post('/login', async (req, res) => {
    console.log("Login route hit");
    const { email, password } = req.body;
    console.log('Login request received:', { email });

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
            res.status(200).json({ 
                message: 'Login successful',
                emailConfirmed: user.confirmed === 1,
                user: { id:user.id, username: user.username, email: user.email}
            });
        });
    });
});

app.post('/submit-report', (req, res) => {
    const { report, id } = req.body;

    console.log('Received report:', { report, id });

    if (!report || !id) {
        return res.status(400).json({ message: 'Report content and user ID are required.' });
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

app.post('/save-constellation', (req, res) => {
    const { userId, name, constellationData } = req.body;
    
    console.log('Received save constellation request:', { userId, name });
    
    if (!userId || !name || !constellationData) {
        return res.status(400).json({ message: 'User ID, name, and constellation data are required.' });
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

app.put('/update-constellation/:id', (req, res) => {
    const constellationId = req.params.id;
    const { name, constellationData } = req.body;

    if (!constellationId || !name || !constellationData) {
        return res.status(400).json({ message: 'Constellation ID, name, and constellation data are required.' });
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


app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});