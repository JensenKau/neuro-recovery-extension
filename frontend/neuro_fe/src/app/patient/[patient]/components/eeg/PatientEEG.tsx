"use client";

import React, { useEffect, useState } from "react";
import PatientEEGItem from "./PatientEEGItem";
import PatientEEGUploadButton from "./PatientEEGUploadButton";
import { Patient, ShortEEG } from "@/app/interface";
import { Typography } from "@mui/material";
import { ToastContainer, toast } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';

interface PatientEEGProps {
	patient: Patient | null;
}

const PatientEEG = ({ patient }: PatientEEGProps) => {
	const [eegs, setEegs] = useState<ShortEEG[]>([]);
	const [uploaded, setUploaded] = useState(false);

	useEffect(() => {
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

		getEEG().then(setEegs);

		if (uploaded) {
			toast.success("Upload Successful", {
				position: "top-center",
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: "light",
			});
		}

		setUploaded(false);
	}, [patient, uploaded]);

	return (
		<div className="flex flex-col gap-5">
			<div className="flex justify-between">
				<div className="my-auto text-3xl text-blue-600">Patient EEGs</div>
				<PatientEEGUploadButton patient={patient} onUpload={() => setUploaded(true)} />
			</div>
			{eegs.length > 0
				? eegs.map((val) => <PatientEEGItem patient={val.patient} label={val.name} datetime={new Date(val.created_at).toLocaleString()} key={val.name} />)
				: <Typography variant="h5" className="mx-auto">This patient currently does not have any EEG uploaded</Typography>
			}
		</div>
	);
};

export default PatientEEG;
