"use client";

import React, {useState} from "react";
import { MenuItem, Select } from "@mui/material";


const BrainEEG = () => {
  const [eegType, setEegType] = useState("raw");

  return (
    <div className="flex flex-col gap-3">
      <Select
        className="w-1/3 ml-auto"
        value={eegType}
        onChange={(e) => setEegType(e.target.value)}
      >
        <MenuItem value="raw">Raw</MenuItem>
        <MenuItem value="adjusted">Adjusted</MenuItem>
        <MenuItem value="cleaned">Cleaned</MenuItem>
      </Select>
      Brain EEG
    </div>
  );
};

export default BrainEEG;