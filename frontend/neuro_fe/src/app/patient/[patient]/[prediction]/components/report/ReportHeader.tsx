"use client";

import { Divider, Typography } from "@mui/material";
import React from "react";
import { Reply } from "@mui/icons-material";
import Link from "next/link";

interface ReportHeaderProps {
  patient: string;
}

const ReportHeader = ({ patient }: ReportHeaderProps) => {
  return (
    <div>
      <div className="flex justify-between">
        <Typography variant="h5" className="my-3 text-5xl">Prediction Report</Typography>
        <Link href={`/patient/${patient}`}>
          <Reply className="text-7xl" />
        </Link>
      </div>
      <Divider />
    </div>
  );
};

export default ReportHeader;