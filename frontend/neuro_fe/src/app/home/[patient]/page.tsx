
import React from 'react'
import HomeIcon from "../../components/HomeIcon";
import WelcomeMessage from "../../components/WelcomeMessage";

const page = () => {
  return (
    <div>
      <HomeIcon />
      <WelcomeMessage header="Welcome back, " name="Ian" />
    </div>
  )
}

export default page
