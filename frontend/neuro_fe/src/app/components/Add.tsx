"use client";
import React from "react";
import Button from "@mui/material/Button";

interface Props {
  children: string;
  icon?: React.ElementType;
  clicked?: () => void;
}

const Add = ({ children, icon: Icon, clicked }: Props) => {
  return (
    <div>
      <Button
        startIcon={Icon ? <Icon /> : undefined}
        variant="contained"
        aria-label="add patient"
        color="primary"
        sx={{ marginRight: "50px" }}
        onClick={clicked}
      >
        {children}
      </Button>
    </div>
  );
};

export default Add;
