import React from "react";

interface Props {
  header: string;
  name: string;
}

const WelcomeMessage = ({header, name}: Props) => {
  return (
    <div
    className="ml-10 mb-10 text-5xl"
    >
      {header} <span className="text-blue-600">{name}</span>
    </div>
  );
};

export default WelcomeMessage;
