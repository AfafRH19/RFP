import React, { useRef, useState, useEffect } from "react";

const FileUpload = ({ setFile, fileType }) => {
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  const [fileName, setFileName] = useState("Aucun fichier sélectionné");
  const [errorMessage, setErrorMessage] = useState("");

  const handleFileSelection = (file) => {
    if (file) {
      setFileName(file.name);
      setFile(file);
      console.log("File received:", file.name);
      console.log("Type:", file.type);
      console.log("Size:", file.size, "bytes");
    } else {
      console.log("No file received.");
    }
  };

  const handleDropEvent = (event) => {
    event.preventDefault();
    dropZoneRef.current.classList.remove("dragover");
    const file = event.dataTransfer.files[0];
    handleFileSelection(file);
  };

  useEffect(() => {
    const dropZoneElement = dropZoneRef.current;
    const fileInputElement = fileInputRef.current;

    const handleDragOver = (event) => {
      event.preventDefault();
      dropZoneElement.classList.add("dragover");
    };

    const handleDragLeave = (event) => {
      event.preventDefault();
      dropZoneElement.classList.remove("dragover");
    };

    const handleDrop = (event) => {
      handleDropEvent(event);
    };

    const handleFileInputClick = () => {
      fileInputElement.click();
    };

    const handleFileChange = (event) => {
      const file = event.target.files[0];
      handleFileSelection(file);
    };

    dropZoneElement.addEventListener("dragover", handleDragOver);
    dropZoneElement.addEventListener("dragleave", handleDragLeave);
    dropZoneElement.addEventListener("drop", handleDrop);
    dropZoneElement.addEventListener("click", handleFileInputClick);
    fileInputElement.addEventListener("change", handleFileChange);

    return () => {
      dropZoneElement.removeEventListener("dragover", handleDragOver);
      dropZoneElement.removeEventListener("dragleave", handleDragLeave);
      dropZoneElement.removeEventListener("drop", handleDrop);
      dropZoneElement.removeEventListener("click", handleFileInputClick);
      fileInputElement.removeEventListener("change", handleFileChange);
    };
  }, []);

  return (
    <div className="file-upload">
      <input
        type="file"
        id="file-input1"
        ref={fileInputRef}
        style={{ display: "none" }}
        name="file1"
        required
      />
      <div id="drop-zone1" ref={dropZoneRef} className="drop-zone">
        {fileType === "document RFP"
          ? "Déposez votre document RFP ici"
          : "Déposez votre documentation produit ici"}
      </div>
      <label htmlFor="file-input1" id="file-label1" className="file-label">
        {fileName}
      </label>
      <div id="error-message1" className="error-message">
        {errorMessage}
      </div>
    </div>
  );
};

export default FileUpload;
