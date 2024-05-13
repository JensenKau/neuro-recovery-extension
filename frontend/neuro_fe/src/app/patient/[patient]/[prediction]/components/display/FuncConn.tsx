"use client";

import { MenuItem, Select } from "@mui/material";
import React, { useState, useEffect } from "react";
import { FunctionalConnectivity } from "@/app/interface";
import FuncConnChart from "./FuncConnChart";

interface FuncConnProps {
  patient_id: number;
  filename: string;
}

const FuncConn = ({ patient_id, filename }: FuncConnProps) => {
  const [fcType, setFcType] = useState("static");
  const [fc, setFc] = useState<FunctionalConnectivity>();

  useEffect(() => {
    const retrieveFc = async () => {
      const res = await fetch(`http://localhost:3000/api/patient_eeg/get_fcs/?patient_id=${patient_id}&filename=${filename}`, {
        method: "GET",
        credentials: "include"
      });
      return await res.json();
    };

    retrieveFc().then(setFc);
  }, [patient_id, filename]);

  return (
    <div className="flex flex-col gap-3">
      <Select
        className="w-1/3 ml-auto"
        value={fcType}
        onChange={(e) => setFcType(e.target.value)}
      >
        <MenuItem value="static">Static</MenuItem>
        <MenuItem value="avg">Dynamic Average</MenuItem>
        <MenuItem value="std">Dynamic Standard Deviation</MenuItem>
      </Select>
      {fc && (
        <>
          <div hidden={fcType !== "static"}>
            <FuncConnChart matrix={fc.static_fc} />
          </div>

          <div hidden={fcType !== "avg"}>
            <FuncConnChart matrix={fc.avg_fc} />
          </div>

          <div hidden={fcType !== "std"}>
            <FuncConnChart matrix={fc.std_fc} />
          </div>
        </>
      )}
    </div>
  )
};

export default FuncConn;