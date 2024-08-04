import os
import fitz
import re

def clean_filename(filename):
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '-')
    return filename

def extract_headings_and_subheadings(pdf_path):
    headings_with_font_size = {}  # Dictionary to store headings with their font sizes

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Traverse each page of the document
    for page_number in range(1, len(pdf_document)):
        page = pdf_document.load_page(page_number)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            lines = block.get("lines", [])
            for i, line in enumerate(lines):
                heading = ""
                line_font_sizes = []  # List to store font sizes of each line
                for span in line.get("spans", []):
                    heading += span["text"]
                    line_font_sizes.append(span["size"])

                # Check if the line is not empty
                if heading.strip():
                    # Extract the font size of the first span
                    font_size = line_font_sizes[0]
                    # Clean up the heading text (remove leading and trailing spaces)
                    heading = heading.strip()
                    # Check if the next line starts with a digit (indicating a new numbered heading)
                    # or if the next line is empty
                    next_line_index = i + 1
                    if next_line_index < len(lines):
                        next_line_text = lines[next_line_index].get("spans", [])
                        if next_line_text:
                            next_line_text = next_line_text[0]["text"]
                            if not re.match(r'\d+\.\s', next_line_text) and not next_line_text.strip():
                                # Append the next line to the current heading
                                heading += " " + next_line_text.strip()
                    # Store the heading with its font size
                    if font_size not in headings_with_font_size:
                        headings_with_font_size[font_size] = [heading]
                    else:
                        headings_with_font_size[font_size].append(heading)

    pdf_document.close()
    return headings_with_font_size

def classify_headings_and_subheadings(font_sizes_with_text):
    # Extract unique font sizes and sort them from largest to smallest
    unique_font_sizes_sorted = sorted(font_sizes_with_text.keys(), reverse=True)

    # Get texts corresponding to the largest and second largest font sizes
    largest_font_size_text = font_sizes_with_text.get(unique_font_sizes_sorted[0], [])
    second_largest_font_size_text = font_sizes_with_text.get(unique_font_sizes_sorted[1], [])

    # Create a list of pairs (text, font size) for headings
    headings = [(text, unique_font_sizes_sorted[0]) for text in largest_font_size_text]

    # Create a list of pairs (text, font size) for subheadings
    subheadings = [(text, unique_font_sizes_sorted[1]) for text in second_largest_font_size_text]

    return headings, subheadings

def add_text_to_pdf(pdf_document, text, font_size):
    # Check if the PDF document has pages
    if len(pdf_document) == 0:
        pdf_document.new_page()  # Create a new page if the document is empty

    # Calculate the available space on the page
    page_width = pdf_document[0].rect.width
    page_height = pdf_document[0].rect.height
    top_margin = 100
    bottom_margin = 100
    available_height = page_height - top_margin - bottom_margin

    # Split the text into lines
    lines = text.split('\n')

    # Initialize variables for current page content
    current_page_content = ""
    current_height = top_margin

    # Get the estimated line height based on font size
    line_height = font_size * 1.2  # Multiplying by 1.2 as an approximation

    # Iterate through each line
    for line in lines:
        # Check if the line fits within the remaining space on the page
        if current_height + line_height < available_height:
            # Add the line to the current page content
            current_page_content += line + '\n'
            current_height += line_height
        else:
            # Add the current page content to the PDF
            page = pdf_document[-1]  # Access the last page
            page.insert_text((10, 100), current_page_content)  # Insert text on the page

            # Create a new page
            pdf_document.new_page()
            current_page_content = line + '\n'
            current_height = top_margin

    # Add the remaining content to the last page
    if current_page_content:
        page = pdf_document[-1]  # Access the last page
        page.insert_text((10, 100), current_page_content)  # Insert text on the page

def create_pdf_for_heading_or_subheading(text, font_size):
    pdf_document = fitz.open()
    add_text_to_pdf(pdf_document, text, font_size)
    return pdf_document

def split_pdf_by_subheadings(pdf_path, subheadings, headings, output_folder):
    # Ensure that the destination folder exists, otherwise create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Initialize variables for content extraction
    subheading_content = ""
    start_extraction = False
    current_subheading = ""
    current_font_size = 0
    last_heading = ""  # Variable to store the last encountered heading

    # Traverse each page of the document
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        blocks = page.get_text("dict")["blocks"]

        # Traverse each block in the page
        for block in blocks:
            lines = block.get("lines", [])
            for line in lines:
                text = ""
                line_font_sizes = []
                for span in line.get("spans", []):
                    text += span["text"]
                    line_font_sizes.append(span["size"])

                # Check if the line is not empty
                if text.strip():
                    # Check if the line contains a subheading
                    if any(subheading_text == text.strip() for subheading_text, _ in subheadings) or any(heading_text == text.strip() for heading_text, _ in headings):
                        # If content extraction has started, save the content to a PDF
                        if start_extraction:
                            subheading_pdf = create_pdf_for_heading_or_subheading(subheading_content, current_font_size)
                            # Name the PDF file with both the subheading and the corresponding heading
                            subheading_pdf_filename = os.path.join(output_folder, f"{clean_filename(last_heading)}_{clean_filename(current_subheading)}.pdf")
                            subheading_pdf.save(subheading_pdf_filename)
                            subheading_pdf.close()
                            print(f"PDF file created for: {current_subheading} -> {subheading_pdf_filename}")

                            # Reset content and extraction flag for the next subheading
                            subheading_content = ""
                            start_extraction = False

                        # Start extraction for the current subheading
                        start_extraction = True
                        current_subheading = text.strip()
                        current_font_size = line_font_sizes[0]  # Assume font size of the first span

                    # If extraction has started, add the line to the content
                    elif start_extraction:
                        subheading_content += text + '\n'

                    # Update the last heading when encountering a heading
                    if any(heading_text == text.strip() for heading_text, _ in headings):
                        last_heading = text.strip()

    # If there is remaining content after reaching the end of the document
    # Save it as a separate PDF file
    if start_extraction:
        subheading_pdf = create_pdf_for_heading_or_subheading(subheading_content, current_font_size)
        # Name the PDF file with both the subheading and the corresponding heading
        subheading_pdf_filename = os.path.join(output_folder, f"{clean_filename(last_heading)}_{clean_filename(current_subheading)}.pdf")
        subheading_pdf.save(subheading_pdf_filename)
        subheading_pdf.close()
        print(f"PDF file created for: {current_subheading} -> {subheading_pdf_filename}")

    pdf_document.close()

# Example function to run the process
def process_pdf(pdf_path, output_folder):
   
    font_sizes_with_text = extract_headings_and_subheadings(pdf_path)
    headings, subheadings = classify_headings_and_subheadings(font_sizes_with_text)
    split_pdf_by_subheadings(pdf_path, subheadings, headings, output_folder)
