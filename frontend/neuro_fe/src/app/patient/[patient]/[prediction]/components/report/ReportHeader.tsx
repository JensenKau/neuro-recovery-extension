"use client";

import { Divider, Typography } from "@mui/material";
import React from "react";

const ReportHeader = () => {
  return (
    <div>
      <Typography variant="h5" className="my-3 text-5xl">Prediction Report</Typography>
      <Divider />      
    </div>
  );
};

export default ReportHeader;