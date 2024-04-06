import React from "react";

interface Props {
  name: string;
}

const WelcomeMessage = ({ name }: Props) => {
  return (
    <div
    className="ml-10 mb-10 text-5xl"
    >
      Welcome Back, <span className="text-blue-600">{name}</span>
    </div>
  );
};

export default WelcomeMessage;
