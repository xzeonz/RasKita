# Enable Frontend and Backend Setup Instructions

## Backend Setup

1. Open a terminal and navigate to the backend directory:
   ```
   cd backend
   ```

2. Create a Python virtual environment (optional but recommended):
   - On Windows:
     ```
     python -m venv venv
     venv\\Scripts\\activate
     ```
   - On macOS/Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Flask backend server:
   ```
   python data/app.py
   ```

   The backend server will start on http://0.0.0.0:5000

## Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install the Node.js dependencies:
   ```
   npm install
   ```

3. Start the frontend development server:
   ```
   npm run dev
   ```

   The frontend will be available at http://localhost:5173 (or the port shown in the terminal).

## Testing the Application

- Open the frontend URL in your browser.
- Upload an image of a cat or dog breed.
- Click the "Predict" button.
- The frontend will send the image to the backend for prediction and display the result.

## Notes

- Ensure the backend server is running before using the frontend.
- The frontend expects the backend API at http://127.0.0.1:5000/predict.
- If you want to deploy or change backend URL, update the API URL in `frontend/src/App.jsx`.
