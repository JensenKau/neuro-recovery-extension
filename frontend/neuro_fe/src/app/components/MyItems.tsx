"use client";
import React from "react";
import { useState } from "react";
import AddIcon from "@mui/icons-material/Add";
import Folder from "@mui/icons-material/Folder";
import AddPatient from "./Add";
import Chips from "./Chips";

const MyItems = () => {
  const [patientNames, setPatientNames] = useState<string[]>([]);
  const addItem = () => {
    setPatientNames(patientNames.concat(["New Patient"]));
  };
  console.log(patientNames);

  return (
    <>
      <div
        style={{
          marginLeft: "45px",
          marginBottom: "20px",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span style={{ color: "blue", fontSize: "30px" }}>My Patients</span>
        <AddPatient icon={AddIcon} clicked={addItem}>
          Add Patient
        </AddPatient>
      </div>

      <Chips items={patientNames} icon={Folder} />
    </>
  );
};

export default MyItems;
