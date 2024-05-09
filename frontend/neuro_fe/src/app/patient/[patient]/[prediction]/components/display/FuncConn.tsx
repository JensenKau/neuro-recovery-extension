"use client";

import { MenuItem, Select } from "@mui/material";
import React, { useState } from "react";

const FuncConn = () => {

  return (
    <div className="flex flex-col gap-3">
      <Select className="w-1/3 ml-auto">
        <MenuItem value="static">Static</MenuItem>
        <MenuItem value="avg">Dynamic Average</MenuItem>
        <MenuItem value="std">Dynamic Standard Deviation</MenuItem>
      </Select>
      FuncConn
    </div>
  )
};

export default FuncConn;