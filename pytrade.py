# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import json
import datetime
from typing import List, Dict, Optional, Tuple, Callable

import pyodbc
import jwt  # PyJWT
from flask import Flask, request, jsonify, Response, make_response
from flask_cors import CORS
import pyodbc



# --- Your modules ---
from analysestock import analysestock
from list import (
    build_companies_payload,
    MARKETS,              # kept for backward compat (not mutated)
    get_markets,          # merged filters (MARKETS + extras)
    search_companies,     # global search helper
)
from assistant import get_answer
from signin import get_db_connection, ensure_user_table_exists  # <- NOTE: from db.py
import bcrypt  # pip install bcrypt
from typing import Tuple
from werkzeug.security import generate_password_hash, check_password_hash

# ------------------------------------------------------------------------------
# App, ENV, CORS
# ------------------------------------------------------------------------------
app = Flask(__name__)

# IMPORTANT: keep this secret strong. You may move it to .env (JWT_SECRET)
app.config["JWT_SECRET"] = os.environ.get(
    "JWT_SECRET",
    "96c63da06374c1bde332516f3acbd23c84f35f90d8a6321a25d790a0a451af32"
)

# Token lifetimes (override in .env if you like)
ACCESS_MINUTES  = int(os.environ.get("ACCESS_MINUTES", "15"))   # 15 minutes
REFRESH_DAYS    = int(os.environ.get("REFRESH_DAYS", "7"))      # 7 days

# Allow your Angular Space + local dev
FRONTEND_ORIGIN = os.environ.get(
    "FRONTEND_ORIGIN",
    "https://pykara-py-trade.static.hf.space,https://localhost:4200"
)
allowed = [o.strip() for o in FRONTEND_ORIGIN.split(",") if o.strip()]
CORS(
    app,
    resources={r"/*": {"origins": allowed}},
    supports_credentials=True,           # allow cookies
    expose_headers=["Authorization"],    # allow SPA to read Authorization if needed
)

# Ensure table exists at startup
ensure_user_table_exists()

# ------------------------------------------------------------------------------
# Helpers: JWT
# ------------------------------------------------------------------------------
def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.timezone.utc)

def _encode_jwt(payload: dict, minutes: Optional[int] = None, days: Optional[int] = None) -> str:
    iat = _now_utc()
    exp = iat
    if minutes:
        exp = iat + datetime.timedelta(minutes=minutes)
    if days:
        exp = iat + datetime.timedelta(days=days)
    to_encode = {
        "iat": int(iat.timestamp()),
        "exp": int(exp.timestamp()),
        **payload,
    }
    return jwt.encode(to_encode, app.config["JWT_SECRET"], algorithm="HS256")

def create_access_token(user_id: int, email: str, name: Optional[str] = None) -> str:
    payload = {"sub": str(user_id), "email": email, "type": "access"}
    if name:
        payload["name"] = name  # optional convenience for UI
    return _encode_jwt(payload, minutes=ACCESS_MINUTES)

def create_refresh_token(user_id: int, email: str) -> str:
    return _encode_jwt({"sub": str(user_id), "email": email, "type": "refresh"}, days=REFRESH_DAYS)

def _decode_jwt(token: str) -> dict:
    return jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])

def set_token_cookies(resp, access_token: str, refresh_token: str):
    """
    Sets HttpOnly cookies. For local dev we use SameSite=None + Secure=False if http.
    In production, set Secure=True and run on HTTPS.
    """
    secure = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
    # Access cookie ~ session/short-lived
    resp.set_cookie(
        "access_token",
        access_token,
        httponly=True,
        secure=secure,
        samesite="None" if secure else "Lax",
        max_age=ACCESS_MINUTES * 60,
        path="/",
    )
    # Refresh cookie ~ longer-lived
    resp.set_cookie(
        "refresh_token",
        refresh_token,
        httponly=True,
        secure=secure,
        samesite="None" if secure else "Lax",
        max_age=REFRESH_DAYS * 24 * 3600,
        path="/refresh",  # scope refresh cookie to the refresh endpoint
    )

def clear_token_cookies(resp):
    resp.delete_cookie("access_token", path="/")
    resp.delete_cookie("refresh_token", path="/refresh")

def extract_bearer_token() -> Optional[str]:
    """
    Reads 'Authorization: Bearer <token>' header if present.
    """
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None

def get_current_user_from_token(require_access: bool = True) -> Optional[dict]:
    """
    Validates either:
      - Bearer access token (preferred), or
      - access_token cookie (fallback).
    Returns dict {userId, email} or None on failure.
    """
    token = extract_bearer_token()
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        return None

    try:
        decoded = _decode_jwt(token)
        if require_access and decoded.get("type") != "access":
            return None
        user_id = decoded.get("sub")
        email = decoded.get("email")
        if not user_id or not email:
            return None
        return {"userId": int(user_id), "email": email}
    except Exception:
        return None

def jwt_required(func: Callable) -> Callable:
    """
    Decorator to protect routes with access token.
    Accepts either Bearer access token or access_token cookie.
    """
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        user = get_current_user_from_token(require_access=True)
        if not user:
            return jsonify({"message": "Unauthorised"}), 401
        # attach to request context if needed
        request.user = user  # type: ignore
        return func(*args, **kwargs)
    return wrapper


def _is_bcrypt_hash(value: str) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith(("$2a$", "$2b$", "$2y$"))

def _safe_check_password(stored_hash: str, password: str) -> Tuple[bool, str]:
    """
    Tries Werkzeug hash first; if it fails, tries bcrypt; then a last-resort
    plain-text compare (for legacy/dev data only).
    Returns (ok, scheme) where scheme is one of "werkzeug", "bcrypt", "plain", "unknown".
    """
    # 1) Werkzeug format (pbkdf2:sha256:...)
    try:
        if check_password_hash(stored_hash, password):
            return True, "werkzeug"
    except Exception:
        pass

    # 2) bcrypt format ($2a/$2b/$2y)
    if _is_bcrypt_hash(stored_hash):
        try:
            if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                return True, "bcrypt"
        except Exception:
            pass

    # 3) plain text (legacy/dev only; not recommended)
    try:
        if stored_hash == password:
            return True, "plain"
    except Exception:
        pass

    return False, "unknown"

def _rehash_and_update_if_needed(user_id: int, email: str, scheme: str, conn) -> None:
    """
    If the user logged in with a legacy hash (bcrypt or plain), immediately
    rehash to the canonical Werkzeug PBKDF2 format and update the DB.
    """
    # Only migrate non-Werkzeug schemes
    if scheme in ("bcrypt", "plain"):
        new_hash = generate_password_hash(email + ":", method="pbkdf2:sha256", salt_length=16)  # temp
        # Above dummy line only to get the *format*; we now re-create with the actual password in _do_signin
        # This function just exists for structure; actual rehash done in _do_signin (see below).
        pass

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}, 200

# ------------------------------------------------------------------------------
# Public APIs
# ------------------------------------------------------------------------------
@app.get("/getfilters")
def get_filters():
    """
    Returns UI filter tree.
    Uses get_markets() so you get MARKETS + extra markets (NASDAQ-100, DAX-40, OMXS-30)
    without changing your original MARKETS object.
    """
    return jsonify({
        "asOf": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "markets": get_markets()
    })

@app.get("/getcompanies")
def get_companies():
    """
    /getcompanies?code=NIFTY50 (or ?index=...)
    """
    code = (request.args.get("code") or request.args.get("index") or "").upper()
    if not code:
        return jsonify({"error": "Missing ?code=<INDEXCODE>"}), 400
    try:
        payload = build_companies_payload(code)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/search_companies")
def route_search_companies():
    """
    /search_companies?q=INFY&indices=NIFTY50,NIFTY100&limit=50
    - q: search term (symbol or company substring)
    - indices: optional CSV of index codes; defaults to all supported
    - limit: 1..200 (default 50)
    """
    q = request.args.get("q", "")
    indices_csv = request.args.get("indices", "")
    limit_raw = request.args.get("limit", "50")

    try:
        limit_i = max(1, min(200, int(limit_raw)))
    except Exception:
        limit_i = 50

    indices = None
    if indices_csv.strip():
        indices = [c.strip().upper() for c in indices_csv.split(",") if c.strip()]

    try:
        results = search_companies(q, indices=indices, limit=limit_i)
        return jsonify({
            "query": q,
            "count": len(results),
            "results": results
        })
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

@app.route('/analysestock', methods=['POST'])
@jwt_required
def analyze_all():
    """
    Protected example. Requires a valid access token (Bearer or cookie).
    """
    try:
        data = request.get_json(force=True)
        tickersSymbol = data['ticker']
        results = []
        for ticker in tickersSymbol:
            try:
                results.append(analysestock(ticker))
            except Exception as e:
                results.append({"ticker": ticker, "error": str(e)})
        return Response(json.dumps(results, indent=2), mimetype='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    try:
        reply = get_answer(user_message)
        return jsonify({"answer": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------------------
# Auth: Sign-up / Sign-in with JWT (no Google)
# ------------------------------------------------------------------------------
@app.route('/signup', methods=['POST'])
def sign_up():
    data = request.json or {}
    name = (data.get('name') or "").strip()
    phone = (data.get('phone') or "").strip()
    email = (data.get('email') or "").strip().lower()
    password = data.get('password') or ""

    if not email or not password:
        return jsonify({"message": "Email and password are required"}), 400

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM Users WHERE email = ?', (email,))
        if cursor.fetchone():
            return jsonify({"message": "Email already in use"}), 400

        # create Werkzeug PBKDF2 hash explicitly
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)

        cursor.execute(
            'INSERT INTO Users (name, phone, email, password) VALUES (?, ?, ?, ?)',
            (name, phone, email, hashed_password)
        )
        conn.commit()
        return jsonify({"message": "User created successfully"}), 201
    finally:
        try: cursor.close()
        except: pass
        conn.close()

def _do_signin(email: str, password: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, email, password FROM Users WHERE email = ?', (email,))
        row = cursor.fetchone()
        if not row:
            return None

        user_id, name, email_db, stored_pw = row[0], row[1], row[2], row[3] or ""

        ok, scheme = _safe_check_password(stored_pw, password)
        if not ok:
            return None

        # If the stored hash was legacy (bcrypt/plain), migrate it now
        if scheme != "werkzeug":
            new_hash = generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)
            try:
                cursor.execute('UPDATE Users SET password = ? WHERE id = ?', (new_hash, user_id))
                conn.commit()
            except Exception:
                # If migration fails, still allow login this time
                pass

        return {"userId": int(user_id), "name": name, "email": email_db}
    finally:
        try:
            cursor.close()
        except:
            pass
        conn.close()


@app.route('/signin', methods=['POST'])
def sign_in():
    """
    Sign in with email + password.
    Returns access & refresh tokens in both:
      - HttpOnly cookies (recommended), and
      - JSON body (for SPA storage if you choose).
    """
    data = request.json or {}
    email = (data.get('email') or "").strip().lower()
    password = data.get('password') or ""

    if not email or not password:
        return jsonify({"message": "Email and password are required"}), 400

    user = _do_signin(email, password)
    if not user:
        return jsonify({"message": "Invalid email or password"}), 401

    access_token = create_access_token(user_id=user["userId"], email=user["email"], name=user["name"])
    refresh_token = create_refresh_token(user_id=user["userId"], email=user["email"])

    payload = {
        "message": "Signed in successfully",
        "userId": user["userId"],
        "name": user["name"],
        "email": user["email"],
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "accessTokenExpiresIn": ACCESS_MINUTES * 60,
        "refreshTokenExpiresIn": REFRESH_DAYS * 24 * 3600,
    }
    resp = make_response(jsonify(payload), 200)
    set_token_cookies(resp, access_token, refresh_token)
    resp.headers["Authorization"] = f"Bearer {access_token}"  # optional
    return resp

@app.post("/refresh")
def refresh():
    """
    Rotate access token using refresh_token cookie (HttpOnly).
    """
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        return jsonify({"message": "Refresh token missing"}), 401

    try:
        decoded = _decode_jwt(refresh_token)
        if decoded.get("type") != "refresh":
            return jsonify({"message": "Invalid token type"}), 401
        user_id = int(decoded.get("sub"))
        email = decoded.get("email")
        if not user_id or not email:
            return jsonify({"message": "Invalid token"}), 401

        new_access = create_access_token(user_id=user_id, email=email)
        payload = {"accessToken": new_access, "accessTokenExpiresIn": ACCESS_MINUTES * 60}
        resp = make_response(jsonify(payload), 200)
        # update access_token cookie only
        secure = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
        resp.set_cookie(
            "access_token",
            new_access,
            httponly=True,
            secure=secure,
            samesite="None" if secure else "Lax",
            max_age=ACCESS_MINUTES * 60,
            path="/",
        )
        resp.headers["Authorization"] = f"Bearer {new_access}"  # optional
        return resp
    except jwt.ExpiredSignatureError:
        return jsonify({"message": "Refresh token expired"}), 401
    except Exception:
        return jsonify({"message": "Invalid refresh token"}), 401

@app.get("/me")
@jwt_required
def me():
    """
    Returns the current user's profile if the access token is valid.
    """
    user = getattr(request, "user", None)
    return jsonify({"userId": user["userId"], "email": user["email"]})

@app.post("/logout")
def logout():
    """
    Clears auth cookies. The client should also forget any in-memory tokens.
    """
    resp = make_response(jsonify({"message": "Logged out"}), 200)
    clear_token_cookies(resp)
    return resp


#community forum to post the data

# --- Add this API anywhere below other routes ---
@app.post("/posts")
@jwt_required
def create_community_post():
    """
    TEMP: Open endpoint. Expects JSON:
    { userId?, userName?, title?, category?, tags?, body }
    """
    data = request.get_json(silent=True) or {}

    user_id = int((data.get("userId") or 0))
    user_name = (data.get("userName") or "").strip() or "Guest"
    title = (data.get("title") or "").strip()
    category = (data.get("category") or "").strip()
    tags = (data.get("tags") or "").strip()
    body = (data.get("body") or "").strip()

    if not body:
        return jsonify({"message": "body is required"}), 400

    conn = get_db_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        # Return the new identity in the same statement (more reliable than SCOPE_IDENTITY())
        cursor.execute("""
            INSERT INTO Community (user_id, user_name, title, category, tags, body)
            OUTPUT INSERTED.id
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, user_name, title, category, tags, body))

        row = cursor.fetchone()
        conn.commit()

        if not row or row[0] is None:
            return jsonify({"error": "Failed to retrieve new post id"}), 500

        new_id = int(row[0])

        return jsonify({
            "id": new_id,
            "message": "Post created",
            "userId": user_id,
            "userName": user_name
        }), 201
    except Exception as e:
        app.logger.exception("create_community_post failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if cursor: cursor.close()
        except:
            pass
        conn.close()


@app.get("/posts")
def list_community_posts():
    """
    List community posts (public). Supports paging:
      GET /posts?limit=50&offset=0
    Returns: { total, offset, limit, count, results: Post[] }
    """
    limit_raw = request.args.get("limit", "50")
    offset_raw = request.args.get("offset", "0")
    try:
        limit = max(1, min(200, int(limit_raw)))
    except Exception:
        limit = 50
    try:
        offset = max(0, int(offset_raw))
    except Exception:
        offset = 0

    conn = get_db_connection()
    cur = None
    try:
        cur = conn.cursor()
        # data page
        cur.execute("""
            SELECT
                id,
                user_id      AS userId,
                user_name    AS userName,
                title,
                category,
                tags,
                body,
                created_at   AS createdAt
            FROM Community
            ORDER BY created_at DESC
            OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
        """, (offset, limit))
        rows = cur.fetchall()

        posts = [
            {
                "id": int(r[0]),
                "userId": int(r[1]),
                "userName": r[2],
                "title": r[3],
                "category": r[4],
                "tags": r[5],
                "body": r[6],
                "createdAt": r[7].isoformat() if hasattr(r[7], "isoformat") else r[7],
            }
            for r in rows
        ]

        # total count
        cur.execute("SELECT COUNT(*) FROM Community")
        total = int(cur.fetchone()[0])

        return jsonify({
            "total": total,
            "offset": offset,
            "limit": limit,
            "count": len(posts),
            "results": posts
        }), 200
    except Exception as e:
        app.logger.exception("list_community_posts failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if cur: cur.close()
        except:
            pass
        conn.close()
# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Default to 5000 locally; on Hugging Face Spaces the platform injects PORT.
    port = int(os.environ.get("PORT", "5000"))
    host = "127.0.0.1" if port == 5000 else "0.0.0.0"
    app.run(host=host, port=port, debug=(host == "127.0.0.1"))
