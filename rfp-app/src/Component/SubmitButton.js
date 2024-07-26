import React, { useState } from "react";

const SubmitButton = ({ file1, file2, description, setResponse }) => {
  const isValid = !!(file1 && file2);
  console.log({ isValid });
  const fetchFiles = async () => {
    const formData = new FormData();
    formData.append("file1", file1);
    formData.append("file2", file2);
    formData.append("description", description);

    try {
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log("Success:", data);
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
