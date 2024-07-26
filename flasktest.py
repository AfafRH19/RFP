# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        files = request.files
        file1 = files.get('file1')
        file2 = files.get('file2')
        description = request.form.get('description')
        print("Description:", description)

        if file1:
            file1.save(os.path.join(UPLOAD_FOLDER, 'RFP_Document.pdf'))
        if file2:
            file2.save(os.path.join(UPLOAD_FOLDER, 'Product_Documentation.pdf'))

        return jsonify({'message': "Files uploaded successfully Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
