'''# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os



app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "http://20.64.144.55:3000"}})  


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        files = request.files
        file = files.get('file')
        file2 = files.get('file2')
        description = request.form.get('description')
        print("Description:", description)

        if file:
            file.save(os.path.join(UPLOAD_FOLDER, 'RFP_Document.pdf'))
        if file2:
            file2.save(os.path.join(UPLOAD_FOLDER, 'Product_Documentation.pdf'))

        

        return jsonify({'message': "Files uploaded successfully ."}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 

'''
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  
UPLOAD_FOLDER = 'uploads'
RFP_FOLDER = os.path.join(UPLOAD_FOLDER, 'RFPDocument')
DOCS_FOLDER = os.path.join(UPLOAD_FOLDER, 'ProductDocs')

# Ensure upload directories exist
for folder in [UPLOAD_FOLDER, RFP_FOLDER, DOCS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        files = request.files
        file = files.get('file')
        docFiles = request.files.getlist('docFiles')  # Use getlist to handle multiple files
        description = request.form.get('description')
        print("Description:", description)

        if file:
            file.save(os.path.join(RFP_FOLDER, 'RFP_Document.pdf'))
        
        if docFiles:
            for i, docFile in enumerate(docFiles, start=1):
                docFile.save(os.path.join(DOCS_FOLDER, f'Product_Documentation_{i}.pdf'))

        response = [
            { "question": "What is your name?", "response": "John Doe" },
            { "question": "What is your age?", "response": "30" },
        ]
        
        return jsonify({'message': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
