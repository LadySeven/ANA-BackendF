# ANA Backend API

This is the FastAPI backend for the ANA Flutter application, providing authentication services with SQLite database.

## Features

- User registration (signup)
- User authentication (login)
- JWT token-based authentication
- Password hashing with bcrypt
- SQLite database storage
- CORS enabled for Flutter app integration

## Setup Instructions

### 1. Install Dependencies

```bash
cd ANA_Backend-master
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### 3. API Documentation

Once the server is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## API Endpoints

### Authentication

- `POST /auth/signup` - Create a new user account
- `POST /auth/login` - Login with username and password
- `GET /auth/me` - Get current user info (requires authentication)

### Example Usage

#### Sign Up
```bash
curl -X POST "http://localhost:8000/auth/signup" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "testuser",
       "email": "test@example.com",
       "password": "password123"
     }'
```

#### Login
```bash
curl -X POST "http://localhost:8000/auth/login" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "testuser",
       "password": "password123"
     }'
```

## Database

The application uses SQLite database (`ana_app.db`) which will be created automatically when you first run the server.

## Security Notes

- Change the `SECRET_KEY` in `auth.py` for production
- Passwords are hashed using bcrypt
- JWT tokens expire after 30 minutes
- CORS is currently set to allow all origins (change for production)

## Flutter Integration

The Flutter app is configured to connect to:
- Android Emulator: `http://10.0.2.2:8000`
- iOS Simulator: `http://localhost:8000`
- Physical Device: Use your computer's IP address

Make sure both the backend server and Flutter app are running for full functionality.
