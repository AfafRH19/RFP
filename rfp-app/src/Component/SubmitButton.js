import React from "react";

const SubmitButton = ({ file, docFiles, description, setResponse }) => {
  const isValid = !!(file && docFiles && docFiles.length > 0);

  const fetchFiles = async () => {
    const formData = new FormData();
    formData.append("file", file);

    // Append each file in the docFiles array with the same key "docFiles"
    docFiles.forEach((docFile) => {
      formData.append("docFiles", docFile);
    });

    formData.append("description", description);

    try {
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResponse(data.message);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <button
      type="button"
      className={`submit-button ${!isValid ? "disabled" : ""}`}
      onClick={fetchFiles}
      disabled={!isValid}
    >
      Submit
    </button>
  );
};

export default SubmitButton;
