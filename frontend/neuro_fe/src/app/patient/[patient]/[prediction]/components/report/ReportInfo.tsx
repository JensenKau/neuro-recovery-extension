"use client";

import React from "react";
import { Typography } from "@mui/material";
import ReportInfoItem from "./ReportInfoItem";

interface ReportInfoProps {
  title: string;
  items: Array<{key: string, value: string}>
}

const ReportInfo = ({title, items}: ReportInfoProps) => {
  return (
    <div className="flex flex-col gap-1">
      <Typography variant="h5" className="text-blue-600 text-3xl">{title}</Typography>
      <div className="grid grid-cols-5">
        {items.map((val) => <ReportInfoItem label={val.key} value={val.value} key={val.key} />)}
      </div>
    </div>
  );
};

export default ReportInfo;