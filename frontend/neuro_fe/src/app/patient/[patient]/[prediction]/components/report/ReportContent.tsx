"use client";

import { Typography, TextField, Button } from "@mui/material";
import { Save, IosShare } from "@mui/icons-material";
import React, { useState } from "react";

interface ReportContentProps {
  eeg_id: number;
  startComment: string;
}

const ReportContent = ({ eeg_id, startComment}: ReportContentProps) => {
  const [comment, setComment] = useState(startComment);

  const saveChanges = async () => {
    const res = await fetch("http://localhost:3000/api/prediction/update_comment", {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        eeg_id: eeg_id,
        comment: comment
      })
    });
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-between">
        <Typography variant="h5" className="text-blue-600 text-3xl">Comment</Typography>
      </div>

      <TextField className="w-full" label="Comment" defaultValue={comment} onChange={(e) => setComment(e.target.value)} maxRows={10} multiline />

      <div className="flex flex-row-reverse gap-2">
        <Button variant="contained" className="flex gap-1 normal-case" onClick={saveChanges}>
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