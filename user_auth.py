import json
from functools import wraps
from flask import redirect, url_for, session, request

USER_DB_FILE = 'users.json'

def load_users():
    try:
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open(USER_DB_FILE, 'w') as f:
        json.dump(users, f)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = password
    save_users(users)
    return True

def authenticate_user(username, password):
    users = load_users()
    return username in users and users[username] == password