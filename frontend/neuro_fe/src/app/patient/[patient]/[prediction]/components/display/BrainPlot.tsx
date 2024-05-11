"use client";

import { MenuItem, Select } from "@mui/material";
import React, { useState } from "react";

interface BrainPlotProps {
  patient_id: number;
  filename: string;
}

const BrainPlot = ({patient_id, filename}: BrainPlotProps) => {
  const [plotType, setPlotType] = useState("static");

  return (
    <div className="flex flex-col gap-3 h-full">
      <Select
        className="w-1/3 ml-auto"
        value={plotType}
        onChange={(e) => setPlotType(e.target.value)}
      >
        <MenuItem value={"static"}>Static</MenuItem>
        <MenuItem value={"dynamic"}>Dynamic Average</MenuItem>
      </Select>
      <iframe
        src={`http://localhost:3000/api/patient_eeg/get_brain_plot/?patient_id=${patient_id}&filename=${filename}&plot=${plotType}`}
        className="h-full"
      />
    </div>
  )
};

export default BrainPlot;