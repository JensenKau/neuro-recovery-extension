import React from "react";

interface Props {
  name: string;
}

const WelcomeMessage = ({ name }: Props) => {
  return (
    <div
      style={{
        marginTop: "10px",
        marginBottom: "30px",
        marginLeft: "40px",
        fontSize: "40px",
      }}
    >
      Welcome Back, <span style={{ color: "blue" }}>{name}</span>
    </div>
  );
};

export default WelcomeMessage;
