"use client";

import React from "react";
import Report from "./components/report/Report";
import Display from "./components/display/Display";

interface PredictionPageProps {
  params: {
    patient: string;
    prediction: string;
  }
}

const page = ({ params }: PredictionPageProps) => {
  return (
    <div className="grid grid-cols-2 h-screen">
      <Display patient_id={parseInt(params.patient)} filename={params.prediction} />
      <Report patient_id={parseInt(params.patient)} filename={params.prediction} />
    </div>
  );
};

export default page;
