import React from "react";
import jsPDF from "jspdf";

const ResponseInput = ({ response }) => {
  const handleClick = () => {
    const doc = new jsPDF();
    const margin = 10; // Set the margin
    const pageHeight = doc.internal.pageSize.height; // Get the page height
    let currentHeight = margin; // Start height tracking with margin

    response.forEach((item, index) => {
      const questionText = `${index + 1}) ${item.question}`;
      const responseText = `- ${item.response}`;

      // Split the question and response texts to ensure they fit within the page width
      const splitQuestionText = doc.splitTextToSize(questionText, 180);
      const splitResponseText = doc.splitTextToSize(responseText, 180);

      // Calculate the height required for the question and response
      const textHeight =
        (splitQuestionText.length + splitResponseText.length) * 10;

      // Check if there is enough space on the current page for both the question and response
      if (currentHeight + textHeight > pageHeight - margin) {
        doc.addPage(); // Add a new page if content won't fit
        currentHeight = margin; // Reset current height for the new page
      }

      // Add the question text
      splitQuestionText.forEach((line) => {
        doc.text(line, margin, currentHeight);
        currentHeight += 10;

        // If the text reaches the bottom of the page, create a new page
        if (currentHeight > pageHeight - margin) {
          doc.addPage();
          currentHeight = margin;
        }
      });

      // Add the response text
      splitResponseText.forEach((line) => {
        doc.text(line, margin, currentHeight);
        currentHeight += 10;

        // If the text reaches the bottom of the page, create a new page
        if (currentHeight > pageHeight - margin) {
          doc.addPage();
          currentHeight = margin;
        }
      });

      // Add extra space after each answer
      currentHeight += 10;
    });

    // Save the PDF file
    doc.save("response.pdf");
  };

  return (
    <button
      className={`submit-button ${!response ? "disabled" : ""}`}
      onClick={handleClick}
      disabled={!response}
    >
      Generate PDF
    </button>
  );
};

export default ResponseInput;
