"use client";

import React, { useEffect, useState } from "react";
import PatientEEGItem from "./PatientEEGItem";
import PatientEEGUploadButton from "./PatientEEGUploadButton";
import { Patient } from "@/app/interface";

interface PatientEEGProps {
	patient: Patient | null;
}

const PatientEEG = ({ patient }: PatientEEGProps) => {
	const [time, setTime] = useState("");

	useEffect(() => {
		setTime(new Date().toLocaleString());
	}, []);

	return (
		<div className="flex flex-col gap-5">
			<div className="flex justify-between">
				<div className="my-auto text-3xl text-blue-600">Patient EEGs</div>
				<PatientEEGUploadButton patient={patient} />
			</div>
			<PatientEEGItem label="slkdfj" datetime={time} />
		</div>
	);
};

export default PatientEEG;
