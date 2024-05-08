"use client";

import React, { useEffect, useState } from "react";
import PatientEEGItem from "./PatientEEGItem";
import PatientEEGUploadButton from "./PatientEEGUploadButton";
import { Patient, ShortEEG } from "@/app/interface";

interface PatientEEGProps {
	patient: Patient | null;
}

const PatientEEG = ({ patient }: PatientEEGProps) => {
	const [eegs, setEegs] = useState<ShortEEG[]>([]);
	const [uploaded, setUploaded] = useState(false);

	const getEEG = async () => {
		if (patient) {
			const res = await fetch(`http://localhost:3000/api/patient_eeg/get_eegs/?patient_id=${patient.id}`, {
				method: "GET",
				credentials: "include"
			});

			return (await res.json()).eegs;
		}

		return [];
	};

	useEffect(() => {
		getEEG().then(setEegs);
		setUploaded(false);
	}, [patient, uploaded]);

	return (
		<div className="flex flex-col gap-5">
			<div className="flex justify-between">
				<div className="my-auto text-3xl text-blue-600">Patient EEGs</div>
				<PatientEEGUploadButton patient={patient} onUpload={() => setUploaded(true)} />
			</div>
			{eegs.map((val) => <PatientEEGItem patient={val.patient} label={val.name} datetime={new Date(val.created_at).toLocaleString()} key={val.name} />)}
		</div>
	);
};

export default PatientEEG;
