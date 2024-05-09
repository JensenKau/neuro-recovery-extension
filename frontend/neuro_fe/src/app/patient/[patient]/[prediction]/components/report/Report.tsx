"use client";

import React, { useState, useEffect } from "react";
import ReportHeader from "./ReportHeader";
import ReportInfo from "./ReportInfo";
import ReportContent from "./ReportContent";
import { EEG, Prediction } from "@/app/interface";

const Report = () => {
  const [eeg, setEeg] = useState<EEG>();
  const [prediction, setPrediction] = useState<Prediction>();

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
    const res = await fetch("http://localhost:3000/api/prediction/get_prediction/?eeg_id=8", {
      method: "GET",
      credentials: "include"
    });

    return await res.json();
  };

  useEffect(() => {
    getEEG().then(setEeg);
    getPrediction().then(setPrediction);
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

        {prediction && <ReportInfo title="Model Prediction" items={[
          { key: "Model ID", value: `${prediction.ai_model.id}` },
          { key: "Model Name", value: prediction.ai_model.name },
          { key: "Outcome Prediction", value: `${prediction.outcome_pred}`},
          { key: "Prediction Confidence", value: `${prediction?.confidence}` }
        ]} />}
      </div>

      <ReportContent />
    </div>
  );
};

export default Report;