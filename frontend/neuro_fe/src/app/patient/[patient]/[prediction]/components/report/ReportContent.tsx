"use client";

import { Typography, TextField, Button } from "@mui/material";
import { Save, IosShare } from "@mui/icons-material";
import React, { useEffect, useState } from "react";
import { User } from "@/app/interface";

interface ReportContentProps {
  eeg_id: number;
  startComment: string;
}

const ReportContent = ({ eeg_id, startComment }: ReportContentProps) => {
  const [comment, setComment] = useState(startComment);
  const [user, setUser] = useState<User>();

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
  };

  const getUser = async () => {
    const res = await fetch("http://localhost:3000/api/user/get_user", {
      method: "GET",
      credentials: "include"
    });
    return await res.json();
  };

  useEffect(() => {
    getUser().then(setUser);
  }, [])

  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-between">
        <Typography variant="h5" className="text-blue-600 text-3xl">Comment</Typography>
      </div>

      {user && user.role === "doctor"
        ? <TextField className="w-full" label="Comment" defaultValue={comment} onChange={(e) => setComment(e.target.value)} maxRows={10} multiline />
        : (comment === ""
          ? <Typography variant="h5" className="text-base mx-auto">No Comment has been Written</Typography>
          : <Typography variant="h5" className="text-base">{comment}</Typography>
        )
      }

      <div className="flex flex-row-reverse gap-2">
        {user && user.role === "doctor" &&
          <Button variant="contained" className="flex gap-1 normal-case" onClick={saveChanges}>
            <Save />
            Save Changes
          </Button>
        }


        <Button variant="contained" className="flex gap-1 normal-case">
          <IosShare />
          Export as PDF
        </Button>
      </div>
    </div>
  );
};

export default ReportContent;