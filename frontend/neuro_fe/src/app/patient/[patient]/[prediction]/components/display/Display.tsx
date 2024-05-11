"use client";

import React, { useState } from "react";
import { Box, Tabs, Tab } from "@mui/material";
import BrainEEG from "./BrainEEG";
import FuncConn from "./FuncConn";
import BrainPlot from "./BrainPlot";

interface DisplayProps {
  patient_id: number;
  filename: string;
}

const Display = ({patient_id, filename}: DisplayProps) => {
  const [tab, setTab] = useState(0);

  return (
    <Box className="m-3 rounded">
      <Tabs value={tab}
        onChange={(e, val) => setTab(val)}
        variant="fullWidth"
        className="rounded my-3"
      >
        <Tab label="Brain EEG" />
        <Tab label="Functional Connectivity" />
        <Tab label="3D Brain Plotting" />
      </Tabs>

      <div hidden={tab !== 0}>
        <BrainEEG patient_id={patient_id} filename={filename} />
      </div>

      <div hidden={tab !== 1}>
        <FuncConn patient_id={patient_id} filename={filename} />
      </div>

      <div hidden={tab !== 2}>
        <BrainPlot />
      </div>
    </Box>
  );
};

export default Display;