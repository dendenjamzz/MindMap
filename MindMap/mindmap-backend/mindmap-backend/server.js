const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const nodemailer = require('nodemailer');
const dotenv = require('dotenv');
const bcrypt = require('bcryptjs');
const mysql = require('mysql2');
const cors = require('cors');
const crypto = require('crypto');
const axios = require('axios');

dotenv.config();

const corsOptions = {
    origin: '*', // Adjust as needed
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Accept'],
};


const app = express();
const PORT = 3002;

// Middleware
app.use(cors(corsOptions));
app.options('*', cors(corsOptions)); // Handle preflight requests

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(express.static(path.join(__dirname, '../../'))); // Serve static files from the root project folder

// Database connection
const db = mysql.createPool({
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
    waitForConnections: true,
    connectionLimit: 10, // You can adjust this number
    queueLimit: 0
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

// Signup route with email check
app.post('/signup', async (req, res) => {
    const { username, email, password } = req.body;
    console.log('Signup request received:', { username, email });

    try {
        // Check if the user already exists
        console.log('Checking if user already exists in the database...');
        db.query('SELECT * FROM users WHERE email = ?', [email], async (err, result) => {
            if (err) {
                console.error('❌ Database error during email check:', err.message);
                console.error('🛠️ Full error object:', err);
                return res.status(500).json({ error: 'Failed to check user', detail: err.message });
            }


            if (result.length > 0) {
                console.log('Email already exists');
                return res.status(400).json({ error: 'Email already exists' });
            }

            // Hash the password
            console.log('Hashing password...');
            const hashedPassword = await bcrypt.hash(password, 10);

            // Insert the new user
            console.log('Inserting new user into the database...');
            db.query(
                'INSERT INTO users (username, email, password, confirmed) VALUES (?, ?, ?, ?)',
                [username, email, hashedPassword, 0],
                (err, result) => {
                    if (err) {
                        console.error('❌ Database error during user insert:', err.message);
                        console.error('🛠️ Full error object:', err);
                        return res.status(500).json({ error: 'Failed to create user', detail: err.message });
                    }


                    // Send confirmation email
                    console.log('Sending confirmation email...');
                    const mailOptions = {
                        from: process.env.EMAIL,
                        to: email,
                        subject: 'MindMap Email Confirmation',
                        html: `
                            <h2>Welcome, ${username}!</h2>
                            <p>Please confirm your email by clicking the link below:</p>
                            <a href="http://localhost:3002/confirm?email=${email}">Confirm Email</a>
                        `,
                    };

                    transporter.sendMail(mailOptions, (error, info) => {
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
    } catch (error) {
        console.error('Unexpected error during signup:', error);
        res.status(500).json({ error: 'An unexpected error occurred' });
    }
});

// Email confirmation route
app.get('/confirm', (req, res) => {
    const { email } = req.query;
    console.log('Email confirmation received for:', email);

    db.query(
        'UPDATE users SET confirmed = 1 WHERE email = ?',
        [email],
        (err, result) => {
            if (err) {
                console.error('Error confirming email:', err);
                return res.status(500).send('Error confirming email');
            }

            if (result.affectedRows === 0) {
                console.log('No user found with the provided email');
                return res.status(400).send('Invalid confirmation link or user does not exist');
            }

            console.log('Email confirmed successfully');
            // Redirect to the login page after confirmation
            res.redirect('/login.html'); // The login page in the root folder
        }
    );
});

app.post('/login', async (req, res) => {
    console.log("Login route hit");  // This log will tell us if the route is being called
    const { email, password } = req.body;
    console.log('Login request received:', { email });

    db.query('SELECT * FROM users WHERE email = ?', [email], async (err, results) => {
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

        // Check if email is confirmed
        console.log('Checking if user is confirmed...');
        if (user.confirmed !== 1) {
            console.log('User has not confirmed email:', email);
            return res.status(400).json({ error: 'Please confirm your email before logging in.' });
        }

        // Verify password using bcrypt
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

app.post('/submit-report', (req, res) => {
    const { report, id } = req.body;

    console.log('Received report:', { report, id });

    if (!report || !id) {
        return res.status(400).json({ message: 'Report content and user ID are required.' });
    }

    // Save the report to the database
    const query = 'INSERT INTO reports (user_id, report_content) VALUES (?, ?)';
    db.query(query, [id, report], (err, result) => {
        if (err) {
            console.error('Error saving the report: ' + err.message);
            return res.status(500).json({ message: 'Error submitting the report.' });
        }

        console.log('Report saved successfully:', result);
        res.status(200).json({ message: 'Report submitted successfully.' });
    });
});

// FOR CONSTELLATION MAKING

const { PythonShell } = require('python-shell');

app.post('/process-words', async (req, res) => {
    console.log('Received POST request to /process-words');
    console.log('Request Body:', req.body);

    const { words } = req.body;
    if (!words) {
        return res.status(400).json({ error: 'No words provided' });
    }

    try {
        // ✅ Send words to Flask API instead of running Python script
        const flaskResponse = await axios.post('http://127.0.0.1:5000/process', { words });

        console.log("Flask response:", flaskResponse.data); // Debugging
        res.json(flaskResponse.data);
    } catch (error) {
        console.error("Error communicating with Flask API:", error);
        res.status(500).json({ error: "Failed to process words. Ensure Flask is running." });
    }
});



// Start server
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});