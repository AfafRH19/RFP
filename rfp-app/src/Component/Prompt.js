import { useState } from "react";

const Prompt = ({ setDescription }) => {
  const [fileDescription, setFileDescription] = useState("");
  const handleDescriptionChange = (event) => {
    setFileDescription(event.target.value);
    setDescription(event.target.value);
  };
  return (
    <div className="input-group">
      <input
        type="text"
        id="file-description"
        name="description"
        placeholder="Enter your prompt"
        value={fileDescription}
        onChange={handleDescriptionChange}
      />
    </div>
  );
};
export default Prompt;
