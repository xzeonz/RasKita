@echo off
REM Template to open backend and frontend servers from the current folder

REM Start backend server
echo Starting backend server...
cd backend
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing backend dependencies...
pip install -r requirements.txt
echo Running backend Flask app...
start cmd /k "python data/app.py"
cd ..

REM Start frontend server
echo Starting frontend server...
cd frontend
echo Installing frontend dependencies...
npm install
echo Running frontend development server...
start cmd /k "npm run dev"
cd ..

echo Both backend and frontend servers are started.
pause
