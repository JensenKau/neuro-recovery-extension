"use client";

import { Typography, TextField, Button } from "@mui/material";
import { Edit, Save, IosShare } from "@mui/icons-material";
import React, { useState } from "react";

const ReportContent = () => {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-between">
        <Typography variant="h5">Comment</Typography>
      </div>

      <TextField className="w-full" label="Comment" maxRows={10} multiline />

      <div className="flex flex-row-reverse gap-2">
        <Button variant="contained" className="flex gap-1 normal-case">
          <Save />
          Save Changes
        </Button>

        <Button variant="contained" className="flex gap-1 normal-case">
          <IosShare />
          Export as PDF
        </Button>
      </div>
    </div>
  );
};

export default ReportContent;