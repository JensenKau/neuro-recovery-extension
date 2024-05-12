"use client";

import {
	Button,
	Dialog,
	DialogContent,
	DialogTitle,
	TextField,
	Typography,
} from "@mui/material";
import React, { useState } from "react";
import UploadFileButton from "./UploadFileButton";
import { Patient } from "@/app/interface";
import { ToastContainer, toast } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';

interface PatientEEGUploadFormProps {
	patient: Patient | null;
	open: boolean;
	onClose(value: boolean): void;
	onUpload(): void;
}

const PatientEEGUploadForm = ({ patient, open, onClose, onUpload }: PatientEEGUploadFormProps) => {
	const [hea, setHea] = useState<File | null>(null);
	const [mat, setMat] = useState<File | null>(null);
	const [filename, setFilename] = useState("");

	const uploadFile = async () => {
		if (patient && hea && mat && filename !== "") {
			const formData = new FormData();
			formData.append("patient_id", `${patient.id}`);
			formData.append("filename", filename);
			formData.append("heaFile", hea);
			formData.append("matFile", mat);

			const res = await fetch("http://localhost:3000/api/patient_eeg/generate_eeg", {
				method: "POST",
				headers: {
					"credentials": "include"
				},
				body: formData
			});

			onUpload();
			onClose(false);
		} else {
			toast.error("Upload Failed", {
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
	};

	return (
		<Dialog
			open={open}
			onClose={() => onClose(false)}
			fullWidth={true}
			maxWidth="sm"
		>
			<DialogTitle className="mx-auto mt-5 mb-3">
				<div>
					<Typography variant="h5" className="text-[#01579b]">
						Upload EEG
					</Typography>
				</div>
			</DialogTitle>

			<DialogContent className="flex flex-col gap-5">
				<TextField className="mt-2" label="File Name" required placeholder="Enter File Name..." onChange={(e) => setFilename(e.target.value)} />
				<UploadFileButton label=".hea" onChange={setHea} />
				<UploadFileButton label=".mat" onChange={setMat} />

				<Button variant="contained" className="w-full" onClick={uploadFile}>
					Upload
				</Button>
			</DialogContent>
		</Dialog>
	);
};

export default PatientEEGUploadForm;
