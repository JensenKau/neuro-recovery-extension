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

const PredictionPage = ({ params }: PredictionPageProps) => {
  return (
    <div className="grid grid-cols-2 h-screen overflow-hidden">
      <Display patient_id={parseInt(params.patient)} filename={params.prediction} />
      <Report patient_id={parseInt(params.patient)} filename={params.prediction} />
    </div>
  );
};

export default PredictionPage;
