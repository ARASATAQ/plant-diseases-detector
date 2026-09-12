"""
Entry point -- starts the FastAPI development server.
Usage:  python run.py
"""
import os
import sys
import uvicorn

# Fix Windows console UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.makedirs('weights',  exist_ok=True)
os.makedirs('uploads',  exist_ok=True)

if __name__ == '__main__':
    print('\n[Plant Disease Detector]')
    print('   Frontend:  http://localhost:8000')
    print('   API docs:  http://localhost:8000/docs\n')
    uvicorn.run(
        'backend.main:app',
        host='0.0.0.0',
        port=8000,
        reload=True,
    )
