import React from "react";
import IconButton from "@mui/material/IconButton";
import HomeButton from "@mui/icons-material/Home";
import Link from "@mui/material/Link";

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
        }}
      >
        <Link href="/home">
          <HomeButton sx={{ fontSize: "70px" }} />
        </Link>
      </IconButton>
    </div>
  );
};

export default HomeIcon;
