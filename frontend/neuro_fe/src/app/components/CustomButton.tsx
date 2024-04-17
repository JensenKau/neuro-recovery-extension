"use client";
import React from "react";
import Button from "@mui/material/Button";

export interface CustomButtonProps {
  children: string;
  icon?: React.ElementType;
  clicked?: () => void;
  style: "contained" | "outlined" | "text";
  buttonHeight?: string;
  buttonWidth?: string;
}

const CustomButton = ({
  children,
  icon: Icon,
  clicked,
  style,
  buttonHeight,
  buttonWidth,
}: CustomButtonProps) => {
  return (
    <div>
      <Button
        startIcon={Icon ? <Icon /> : undefined}
        variant={style}
        aria-label="add patient"
        color="primary"
        onClick={clicked}
        sx={{
          width: buttonWidth === undefined ? undefined : buttonWidth,
          height: buttonHeight === undefined ? undefined : buttonHeight,
        }}
      >
        {children}
      </Button>
    </div>
  );
};

export default CustomButton;
