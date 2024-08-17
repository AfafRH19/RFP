import React, { useRef, useState, useEffect } from "react";

const FileUpload = ({ setFile, fileType, multiple = false }) => {
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  const [fileNames, setFileNames] = useState(
    multiple ? [] : "Aucun fichier sélectionné *"
  );
  const [errorMessage, setErrorMessage] = useState("");

  const handleFileSelection = (files) => {
    if (files && files.length > 0) {
      if (multiple && fileType !== "document RFP") {
        const newFiles = Array.from(files);
        const updatedFileNames = [
          ...fileNames,
          ...newFiles.map((file) => file.name),
        ];
        const updatedFiles = [
          ...(Array.isArray(fileNames) ? fileNames : []),
          ...newFiles,
        ];

        setFileNames(updatedFileNames);
        setFile(updatedFiles);

        newFiles.forEach((file) => {
          console.log("File added:", file.name);
          console.log("Type:", file.type);
          console.log("Size:", file.size, "bytes");
        });
      } else {
        const file = files[0];
        setFileNames(file.name);
        setFile(file);

        console.log("File received:", file.name);
        console.log("Type:", file.type);
        console.log("Size:", file.size, "bytes");
      }
    } else {
      setFileNames(multiple ? [] : "Aucun fichier sélectionné *");
      console.log("No file received.");
    }
  };

  const handleDropEvent = (event) => {
    event.preventDefault();
    dropZoneRef.current.classList.remove("dragover");
    const files = event.dataTransfer.files;
    handleFileSelection(files);
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
      const files = event.target.files;
      handleFileSelection(files);
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
  }, [multiple, fileNames, fileType]);

  return (
    <div className="file-upload">
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: "none" }}
        multiple={multiple}
        required
      />
      <div ref={dropZoneRef} className="drop-zone">
        {fileType === "document RFP"
          ? "Déposez votre document RFP ici"
          : "Déposez votre documentation produit ici"}
      </div>
      {multiple ? (
        fileNames.length === 0 ? (
          <label className="file-label">Aucun fichier sélectionné *</label>
        ) : (
          <>
            {fileNames.slice(0, 3).map((fileName, index) => (
              <label key={index} className="file-label">
                {fileName}
              </label>
            ))}
            {fileNames.length > 3 && (
              <label className="file-label">+{fileNames.length - 3}</label>
            )}
          </>
        )
      ) : (
        <label className="file-label">{fileNames}</label>
      )}
      <div className="error-message">{errorMessage}</div>
    </div>
  );
};

export default FileUpload;
