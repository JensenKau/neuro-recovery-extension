"use client";

import { Typography } from "@mui/material";
import React from "react";

interface ReportInfoItemProps {
  label: string;
  value: string;
}

const ReportInfoItem = ({label, value}: ReportInfoItemProps) => {
  return (
    <>
      <Typography variant="h5" className="col-span-3 text-lg font-bold">{label}</Typography>
      <Typography variant="h5" className="col-span-2 text-lg">{value}</Typography>
    </>
  );
};

export default ReportInfoItem;