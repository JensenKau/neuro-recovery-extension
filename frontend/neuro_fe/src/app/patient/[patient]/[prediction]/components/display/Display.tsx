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
    <Box className="m-3 rounded h-full">
      <Tabs value={tab}
        onChange={(e, val) => setTab(val)}
        variant="fullWidth"
        className="rounded my-3"
      >
        <Tab label="Brain EEG" />
        <Tab label="Functional Connectivity" />
        <Tab label="3D Brain Plotting" />
      </Tabs>

      <div hidden={tab !== 0} className="h-full">
        <BrainEEG patient_id={patient_id} filename={filename} />
      </div>

      <div hidden={tab !== 1} className="h-full">
        <FuncConn patient_id={patient_id} filename={filename} />
      </div>

      <div hidden={tab !== 2} className="h-full">
        <BrainPlot patient_id={patient_id} filename={filename} />
      </div>
    </Box>
  );
};

export default Display;