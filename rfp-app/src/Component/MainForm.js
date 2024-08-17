import { useState } from "react";
import FileUpload from "./FileUpload";
import Prompt from "./Prompt";
import SubmitButton from "./SubmitButton";
import ResponseInput from "./ResponseInput";

const MainForm = () => {
  const [file, setFile] = useState(null);
  const [docFiles, setDocFiles] = useState(null);
  const [response, setResponse] = useState(null);
  const [description, setDescription] = useState("");

  return (
    <div className="container">
      <div className="form-group">
        <div className="file-uploads">
          <FileUpload setFile={setFile} fileType="document RFP" />
          <FileUpload
            setFile={setDocFiles}
            fileType="product documentation"
            multiple={true}
          />
        </div>
        <Prompt setDescription={setDescription} />
        <div className="buttons">
          <SubmitButton
            file={file}
            docFiles={docFiles}
            description={description}
            setResponse={setResponse}
          />
          <ResponseInput response={response} />
        </div>
      </div>
    </div>
  );
};

export default MainForm;
