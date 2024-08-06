from QuestionAnswer import process_pdfs_and_ask_questions
from Split1 import process_pdf
from RequirementEx import requirement_extraction

def main():
    process_pdf( pdf_path = "/home/innov_user/ModelQT/test/RFP-main2/uploads/RFP_Document.pdf",output_folder="/home/innov_user/ModelQT/test/final/SplitedDocument/")
    requirement_extraction(pdf_dir_path = "/home/innov_user/ModelQT/test/final/SplitedDocument/", result_dir_path = "/home/innov_user/ModelQT/test/final/requirementExtraction/")
    process_pdfs_and_ask_questions()
  

if "__name__" == "__main__":
    main()