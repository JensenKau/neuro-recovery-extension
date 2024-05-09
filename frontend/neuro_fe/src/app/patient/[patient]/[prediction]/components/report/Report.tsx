"use client";

import React, { useState, useEffect } from "react";
import ReportHeader from "./ReportHeader";
import ReportInfo from "./ReportInfo";
import ReportContent from "./ReportContent";
import { EEG } from "@/app/interface";

const Report = () => {
  const [eeg, setEeg] = useState<EEG>();

  const getTime = (time: number) => {
    const hr = Math.floor(time / 3600);
    const min = Math.floor(time / 60) % 60;
    const sec = time % 60;

    const hrStr = hr.toLocaleString('en-US', { minimumIntegerDigits: 2, useGrouping: false });
    const minStr = min.toLocaleString('en-US', { minimumIntegerDigits: 2, useGrouping: false });
    const secStr = sec.toLocaleString('en-US', { minimumIntegerDigits: 2, useGrouping: false });

    return `${hrStr}:${minStr}:${secStr}`
  }

  const getEEG = async () => {
    const res = await fetch("http://localhost:3000/api/patient_eeg/get_eeg/?patient_id=4&filename=some_file_name", {
      method: "GET",
      credentials: "include"
    });

    return await res.json();
  };

  const getPrediction = async () => {

  };

  useEffect(() => {
    getEEG().then(setEeg);
  }, [])

  return (
    <div className="flex flex-col gap-5 bg-slate-200 h-full py-3 px-5">
      <ReportHeader />

      <div className="grid grid-cols-2">
        {eeg && <ReportInfo title="EEG Information" items={[
          { key: "Start Time", value: getTime(eeg.start_time) },
          { key: "End Time", value: getTime(eeg.end_time) },
          { key: "Utility Frequency", value: `${eeg.utility_freq}` },
          { key: "Sampling Frequency", value: `${eeg.sampling_freq}` }
        ]} />}
        

        <ReportInfo title="Model Prediction" items={[
          { key: "Model Name", value: getTime(0) },
          { key: "Outcome Prediction", value: getTime(0) },
          { key: "Prediction Confidence", value: `${0}` },
        ]} />
      </div>

      <ReportContent />
    </div>
  );
};

export default Report;