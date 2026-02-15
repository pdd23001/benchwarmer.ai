import sqlite3
import json
import os
from typing import List, Dict, Any, Optional

# Resolve DB path relative to the project root (two levels up from this file)
# This ensures the same chat.db is used regardless of where the server is started from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
DB_PATH = os.path.join(_PROJECT_ROOT, "chat.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    # Create messages table
    # session_id, role, content, tools (json), plot_image (path/base64?), timestamp
    conn.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tools TEXT,
            plot_image TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS algorithms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            code TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str, tools: Optional[List[Dict]] = None, plot_image: Optional[str] = None):
    conn = get_db_connection()
    tools_json = json.dumps(tools) if tools else None
    conn.execute(
        'INSERT INTO messages (session_id, role, content, tools, plot_image) VALUES (?, ?, ?, ?, ?)',
        (session_id, role, content, tools_json, plot_image)
    )
    conn.commit()
    conn.close()

def get_messages(session_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC',
        (session_id,)
    ).fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        msg = {
            "id": str(row["id"]), # Convert integer ID to string for frontend consistency
            "role": row["role"],
            "content": row["content"],
            "tools": json.loads(row["tools"]) if row["tools"] else [],
            "plotImage": row["plot_image"]
        }
        messages.append(msg)
    return messages

def get_all_sessions() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    # Ensure session_titles table exists
    conn.execute('''
        CREATE TABLE IF NOT EXISTS session_titles (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL
        )
    ''')
    rows = conn.execute('''
        SELECT m1.session_id, MIN(m1.timestamp) as created_at,
               (SELECT content FROM messages m2 WHERE m2.session_id = m1.session_id ORDER BY id ASC LIMIT 1) as first_message,
               st.title as custom_title
        FROM messages m1
        LEFT JOIN session_titles st ON st.session_id = m1.session_id
        GROUP BY m1.session_id
        ORDER BY created_at DESC
    ''').fetchall()
    conn.close()

    sessions = []
    for row in rows:
        if row["custom_title"]:
            title = row["custom_title"]
        elif row["first_message"]:
            title = (row["first_message"][:30] + "...") if len(row["first_message"]) > 30 else row["first_message"]
        else:
            title = "New Chat"
        sessions.append({
            "id": row["session_id"],
            "created_at": row["created_at"],
            "title": title,
        })
    return sessions

def delete_session(session_id: str):
    conn = get_db_connection()
    conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
    conn.execute('DELETE FROM session_titles WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()

def rename_session(session_id: str, title: str):
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS session_titles (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL
        )
    ''')
    conn.execute(
        'INSERT OR REPLACE INTO session_titles (session_id, title) VALUES (?, ?)',
        (session_id, title)
    )
    conn.commit()
    conn.close()

def get_session_title(session_id: str) -> Optional[str]:
    conn = get_db_connection()
    # Ensure table exists
    conn.execute('''
        CREATE TABLE IF NOT EXISTS session_titles (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL
        )
    ''')
    row = conn.execute('SELECT title FROM session_titles WHERE session_id = ?', (session_id,)).fetchone()
    conn.close()
    return row["title"] if row else None

def save_algorithm(name: str, code: str):
    conn = get_db_connection()
    conn.execute(
        'INSERT OR REPLACE INTO algorithms (name, code) VALUES (?, ?)',
        (name, code)
    )
    conn.commit()
    conn.close()

def get_all_algorithms() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute('SELECT name, created_at FROM algorithms ORDER BY created_at DESC').fetchall()
    conn.close()
    return [{"name": row["name"], "created_at": row["created_at"]} for row in rows]

def get_algorithm_code(name: str) -> Optional[str]:
    conn = get_db_connection()
    row = conn.execute('SELECT code FROM algorithms WHERE name = ?', (name,)).fetchone()
    conn.close()
    return row["code"] if row else None
