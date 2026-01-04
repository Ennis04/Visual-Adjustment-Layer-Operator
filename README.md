Step to run the program

1. cd to backend and run this line on command prompt
   py -3.12 -m venv venv

2. you should see a new venv folder inside backend, then run
   .\venv\Scripts\Activate.ps1

3. To run this file, please make sure that your python version is 3.12.x, check the version with
   python --version

4. Install the needed library
   python -m pip install --upgrade pip setuptools wheel
   pip install fastapi uvicorn pydantic numpy opencv-python

5. Check to see whether numpy and cv2 are installed successfully
   python -c "import numpy as np; import cv2; print('numpy', np.__version__); print('cv2', cv2.__version__)"

6. Run this to start the backend
   uvicorn server:app --reload --host 127.0.0.1 --port 8000

7. Ctrl + click on the link to diret to browser or type "http://127.0.0.1:8000" at browser
   
8. Open VS code and run "home.html" using live server.
