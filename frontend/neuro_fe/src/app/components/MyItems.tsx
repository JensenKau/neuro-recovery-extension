"use client";
import React, {ComponentType} from "react";
import { useState } from "react";
import Folder from "@mui/icons-material/Folder";
import Chips from "./Chips";
import {FormProps} from "./PatientForm"

interface Props{
  children: string;
  initialItems: string[];
  FormButtonComponent?: ComponentType<FormProps>;
  FormButtonProps?: FormProps;
}

const MyItems = ({children, initialItems, FormButtonComponent, FormButtonProps}: Props) => {
  const [items, setItems] = useState<string[]>(initialItems);
  const handleFormSubmit = (email: string) => {
    setItems(currentItems => [...currentItems, email]);
  };

  return (
    <div style={{marginBottom: "65px"}}>
      <div
        style={{
          marginLeft: "45px",
          marginBottom: "20px",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span style={{ color: "blue", fontSize: "30px" }}>{children}</span>
        {FormButtonComponent === undefined || FormButtonProps === undefined? <span></span>: <FormButtonComponent {...FormButtonProps} onSubmit={handleFormSubmit}/>}
      </div>
      
      <Chips items={items} icon={Folder} />
    </div>
  );
};

export default MyItems;
