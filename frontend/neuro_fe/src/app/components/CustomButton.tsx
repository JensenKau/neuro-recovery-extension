"use client";
import React from "react";
import Button from "@mui/material/Button";

export interface CustomButtonProps {
  children: string;
  icon?: React.ElementType;
  clicked?: () => void;
}

const CustomButton = ({ children, icon: Icon, clicked }: CustomButtonProps) => {
  return (
    <div>
      <Button
        startIcon={Icon ? <Icon /> : undefined}
        variant="outlined"
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

export default CustomButton;