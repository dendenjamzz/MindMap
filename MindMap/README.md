# MindMap

MindMap is a web-based mind-mapping application with email-confirmed accounts, MySQL persistence, and a Flask NLP microservice that generates semantically related word suggestions using WordNet and Wu-Palmer similarity scoring.

## Stack

- **Frontend:** static HTML/CSS/vanilla JS served by Express
- **Backend:** Node.js/Express — authentication, constellation persistence, report management
- **NLP microservice:** Flask (Python) — semantic expansion, word definitions, content moderation
- **Database:** MySQL — three tables: `users`, `constellations`, `reports`

## Setup

1. Requirements: Node 18+, Python 3.10+, MySQL 8+.
2. Copy `mindmap-backend/.env.example` to `mindmap-backend/.env` and fill in the values.
3. Install Node dependencies: `cd mindmap-backend && npm install`.
4. Create and activate a Python virtual environment, then install Flask dependencies: `pip install flask flask-cors nltk googletrans==4.0.0rc1`.
5. On first run, download the NLTK WordNet corpus: open a Python shell and run `import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')`.
6. Start the Flask microservice: `python flask_server.py` inside `mindmap-backend`.
7. Start the Node backend: `npm start` inside `mindmap-backend`.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `DB_HOST` | MySQL host |
| `DB_USER` | MySQL user |
| `DB_PASSWORD` | MySQL password |
| `DB_NAME` | MySQL database name |
| `EMAIL` | Sender address for confirmation emails |
| `EMAIL_PASSWORD` | SMTP password |
| `APP_URL` | Public URL used in confirmation email links |
| `FRONTEND_ORIGIN` | Comma-separated allowed CORS origins |
| `FLASK_URL` | Flask service base URL (e.g. `http://127.0.0.1:5000`) |
| `JWT_SECRET` | Strong random string for signing JWT tokens — never commit this value |

## Notes

- Database tables (`users`, `constellations`, `reports`) auto-provision on startup via `CREATE TABLE IF NOT EXISTS`.
- Email confirmation is required before login succeeds.
- JWT tokens are issued on login and must be sent as `Authorization: Bearer <token>` for all protected endpoints.
- Rate limiting is applied via `express-rate-limit` (100 requests / 15 min per IP on general routes).
- Security headers are set via Helmet.
- The Flask service must be running before the Node backend starts; `/process-words` proxies to it.

## User Guide

### Sign Up

1. Open `signup.html` in your browser.
2. Fill in your username, email, and password.
3. Click **Submit**.
4. Check your email and click the confirmation link before logging in.

### Login

1. Open `login.html` in your browser.
2. Enter your email and password.
3. Click **Submit**.
4. On success you are redirected to your account dashboard.

### Start a New Constellation

1. From the dashboard click **Start a New Constellation**.
2. Enter up to 10 words separated by commas (e.g. `cat, dog, apple`).
3. Click **Submit**.
4. The system generates a force-directed constellation with nodes and weighted links.

### Interact with Your Constellation

- **Drag nodes** — click and drag any node to reposition it.
- **Zoom** — use the scroll wheel or on-screen zoom controls.
- **Add related words** — click any node to open its popup, then click a suggestion to add it as a new connected node.
- **See a word definition** — in the node popup click **Meaning** to display a short WordNet definition.
- **Delete a node** — in the node popup click **Delete node** to remove that node and all its edges.

### Save Your Work

Click **Save Constellation** in the editor. The map is stored in MySQL and linked to your account.

### Managing Saved Constellations

From the dashboard each saved constellation has four buttons:

| Button | Action |
|--------|--------|
| **Edit** | Load the constellation back into the editor |
| **Delete** | Permanently remove the constellation |
| **PNG** | Download a PNG image of the constellation |
| **CSV Report** | Download a CSV file of all nodes and links |

### Submit a Report

Click the **Report** button in the dashboard left sidebar. A text area appears — write your feedback and click **Submit Report**.

### Admin: Manage Reports *(admin accounts only)*

Click **Manage Reports** in the dashboard sidebar. The admin panel lists all user reports with author, content, status, and date. Use the filter buttons (All / Pending / Reviewed / Closed) to narrow the list, then update any report's status via the inline dropdown and click **Save**.

## License

MIT
