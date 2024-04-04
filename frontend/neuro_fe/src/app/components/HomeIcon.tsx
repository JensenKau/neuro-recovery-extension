import React from "react";
import IconButton from "@mui/material/IconButton";
import HomeButton from "@mui/icons-material/Home";

const HomeIcon = () => {
  return (
    <div>
      <IconButton
        aria-label="home"
        color="primary"
        sx={{
          ":hover": {
            backgroundColor: "#d1d5db",
          },
          margin: "30px"
        }}
      >
        <HomeButton sx={{ fontSize: "50px" }} />
      </IconButton>
    </div>
  );
};

export default HomeIcon;
