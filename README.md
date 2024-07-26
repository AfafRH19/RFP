# RFP
____
## Start the project

1- Clone the project: git@github.com:AfafRH19/RFP.git
2- to run the flask : 

rm -rf venv  
python3 -m venv venv  
source venv/bin/activate

Pip install Flask flask-cors
Pip install Flask
pip install pymupdf
Python3 flasktest.py  


3- to run ReactJS:

cd rfp-app
npm install
npm start 


## Projet Architecture

- rfp-app: front of the project (REACT)
- Uploads: It's the directory that groups the documents sent by the front-end. It will overwrite the existing documents every time new documents are sent, and they are automatically renamed to avoid confusion.
- flasktest.py : it s main of the flask (endpoint that receive the docs and send response)
