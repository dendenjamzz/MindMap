# MindMap

MindMap is a lightweight mind-mapping app with email-confirmed accounts, MySQL persistence, and a Flask-based NLP helper for generating node suggestions.

## Stack
- Frontend: static HTML/CSS/JS served by Express
- Backend: Node/Express with MySQL and Nodemailer
- NLP helper: Flask service consumed by `/process-words`

## Setup
1. Requirements: Node 18+, Python 3.10+, MySQL 8+.
2. Copy `mindmap-backend/mindmap-backend/.env.example` to `.env` and fill values.
3. Install backend deps: `cd mindmap-backend/mindmap-backend && npm install`.
4. Install Python deps (for Flask): activate your venv, then `pip install -r requirements.txt` (if present) or ensure Flask + needed libs are installed.
5. Run Flask helper: `python flask_server.py` in `mindmap-backend/mindmap-backend`.
6. Run Node backend: `npm start` in `mindmap-backend/mindmap-backend`.

## Environment variables (backend)
- `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- `EMAIL`, `EMAIL_PASSWORD`
- `APP_URL` (public URL for confirmation links)
- `FRONTEND_ORIGIN` (comma-separated allowed origins for CORS)
- `FLASK_URL` (e.g., `http://127.0.0.1:5000/process`)

## Notes
- Database tables auto-provision on startup.
- Email confirmation is required before login succeeds.
- `/process-words` proxies to the Flask service; keep it running.

## License
MIT
