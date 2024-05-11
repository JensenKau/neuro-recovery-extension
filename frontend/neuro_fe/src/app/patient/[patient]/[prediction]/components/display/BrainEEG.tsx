"use client";

import { MenuItem, Select } from "@mui/material";
import React, { useState, useEffect } from "react";

interface BrainEEGProps {
  patient_id: number;
  filename: string;
}

const BrainEEG = ({ patient_id, filename }: BrainEEGProps) => {
  const regions = [
    "Fp1", "Fp2", "F7", "F8", "F3", "F4", "T3", "T4", "C3", "C4", "T5",
    "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "Fpz", "Oz", "F9"
  ];

  const [points, setPoints] = useState<Array<Array<number>>>();
  const [Chart, setChart] = useState<any>();
  const [region, setRegion] = useState(0);

  const getEegPoints = async () => {
    const res = await fetch(`http://localhost:3000/api/patient_eeg/get_eeg_points/?patient_id=${patient_id}&filename=${filename}`, {
      method: "GET",
      credentials: "include"
    });
    return (await res.json()).data;
  };

  const chartProperty = ((regionIndex: number) => {
    if (points) {
      const acceptedPoints = Math.ceil(points[0].length / 750);
      const series = {
        name: regions[regionIndex],
        data: points[regionIndex].filter((val, index) => index % acceptedPoints === 0).map((val, index) => { return { x: index, y: val }; })
      };

      return {
        chart: { type: 'line' },
        series: [series],
        xaxis: { tickAmount: 20 },
        yaxis: {
          min: -1,
          max: 1,
          labels: { formatter: (value: number) => Math.round((value + Number.EPSILON) * 100) / 100 }
        },
        stroke: { width: 1 }
      }
    }
  })(region);

  useEffect(() => {
    getEegPoints().then(setPoints);
    import("react-apexcharts").then((mod) => {
      setChart(() => mod.default);
    });
  }, []);

  return (
    <div className="flex flex-col gap-3">
      <Select
        className="w-1/3 ml-auto"
        value={region}
        onChange={(e) => {
          if (typeof e.target.value === "number") {
            setRegion(e.target.value);
          }
        }}
      >
        {regions.map((region, index) => <MenuItem value={index} key={region}>{region}</MenuItem>)}
      </Select>
      {points && Chart && <Chart options={chartProperty} series={chartProperty?.series} type="line" />}
    </div>
  );
};

export default BrainEEG;