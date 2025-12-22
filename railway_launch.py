#!/usr/bin/env python3
"""
Railway Launcher - Guaranteed PORT handling
"""
import os
import sys
import subprocess

def main():
    # Get PORT from environment, default to 5000
    port_str = os.environ.get('PORT', '5000').strip()
    
    # Clean the PORT value
    # Remove any literal "$PORT" string
    if port_str == '$PORT':
        print("WARNING: Found literal '$PORT', using default 5000")
        port = 5000
    else:
        try:
            port = int(port_str)
        except ValueError:
            print(f"ERROR: Invalid PORT '{port_str}', using 5000")
            port = 5000
    
    print(f"🚀 Starting Pytrade Backend on port: {port}")
    print(f"📦 Environment: PORT={port_str}, Resolved={port}")
    
    # Build the gunicorn command
    cmd = [
        'gunicorn',
        '--bind', f'0.0.0.0:{port}',
        '--access-logfile', '-',
        '--error-logfile', '-',
        'pytrade:app'
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    
    # Execute gunicorn (replace current process)
    os.execvp('gunicorn', cmd)

if __name__ == '__main__':
    main()