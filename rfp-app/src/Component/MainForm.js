import { useState } from "react";
import FileUpload from "./FileUpload";
import Prompt from "./Prompt";
import SubmitButton from "./SubmitButton";
import ResponseInput from "./ResponseInput";

const MainForm = () => {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [response, setResponse] = useState("");
  const [description, setDescription] = useState("");

  console.log(response);
  return (
    <div className="container">
      <div className="form-group">
        <FileUpload setFile={setFile1} fileType="document RFP" />
        <FileUpload setFile={setFile2} fileType="product documentation" />
        <Prompt setDescription={setDescription} />
        <SubmitButton
          file1={file1}
          file2={file2}
          description={description}
          setResponse={setResponse}
        />
      </div>
      <ResponseInput response={response} />
    </div>
  );
};

export default MainForm;
