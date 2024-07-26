import React, { useState, useEffect } from "react";

const ResponseInput = ({ response }) => {
  const [displayedText, setDisplayedText] = useState("");

  useEffect(() => {
    let index = 0;
    const typingInterval = 50; // Time in milliseconds between each letter

    // Function to handle typing effect
    const typeText = () => {
      if (index < response.length) {
        setDisplayedText((prev) => prev + response[index]);
        index++;
      } else {
        clearInterval(intervalId); // Stop the interval once the entire text is typed
      }
    };

    // Set interval to type text
    const intervalId = setInterval(typeText, typingInterval);

    // Cleanup function to clear the interval when component unmounts or response changes
    return () => {
      clearInterval(intervalId);
    };
  }, [response]);

  return (
    <textarea
      type="text"
      id="server-response"
      placeholder="Réponse du serveur"
      value={displayedText}
      readOnly
    />
  );
};

export default ResponseInput;
