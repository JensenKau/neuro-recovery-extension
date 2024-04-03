import React from "react";
import HomeIcon from "../components/HomeIcon";
import WelcomeMessage from "../components/WelcomeMessage";
import MyPatients from "../components/MyItems";

const page = () => {
  return (
    <div>
      <HomeIcon />
      <WelcomeMessage name='Ian'/>
      <MyPatients />
    </div>
  );
};

export default page;
