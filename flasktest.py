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

        process_pdf( pdf_path = "/home/innov_user/ModelQT/test/RFP-main2/uploads/RFP_Document.pdf",output_folder="/home/innov_user/ModelQT/test/final/SplitedDocument/")
        requirement_extraction(pdf_dir_path = "/home/innov_user/ModelQT/test/final/SplitedDocument/", result_dir_path = "/home/innov_user/ModelQT/test/final/requirementExtraction/", custom_template=description)
        response = process_pdfs_and_ask_questions()

        return jsonify({'message': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)