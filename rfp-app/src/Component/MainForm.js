import { useState } from "react";
import FileUpload from "./FileUpload";
import Prompt from "./Prompt";
import SubmitButton from "./SubmitButton";
import ResponseInput from "./ResponseInput";

const MainForm = () => {
  const [file, setFile] = useState(null);
  const [docFiles, setDocFiles] = useState(null);
  const [response, setResponse] = useState("");
  const [description, setDescription] = useState("");

  return (
    <div className="container">
      <div className="form-group">
        <FileUpload setFile={setFile} fileType="document RFP" />
        <FileUpload
          setFile={setDocFiles}
          fileType="product documentation"
          multiple={true}
        />
        <Prompt setDescription={setDescription} />
        <SubmitButton
          file={file}
          docFiles={docFiles}
          description={description}
          setResponse={setResponse}
        />
      </div>
      <ResponseInput response={response} />
    </div>
  );
};

export default MainForm;
