"use client";

import {
	Button,
	Dialog,
	DialogContent,
	DialogTitle,
	Typography,
} from "@mui/material";
import React, { useState } from "react";
import UploadFileButton from "./UploadFileButton";

interface PatientEEGUploadFormProps {
	open: boolean;
	onClose(value: boolean): void;
}

const PatientEEGUploadForm = ({ open, onClose }: PatientEEGUploadFormProps) => {
	const [hea, setHea] = useState<File | null>(null);
	const [mat, setMat] = useState<File | null>(null);

	const uploadFile = async () => {
		if (hea && mat) {
			const formData = new FormData();
			formData.append("heaFile", hea);
			formData.append("matFile", mat);

			const res = await fetch("http://localhost:3000/api/patient_eeg/generate_eeg", {
				method: "POST",
				headers: {
					"credentials": "include"
				},
				body: formData
			});

			onClose(false);
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
						New Patient
					</Typography>
				</div>
			</DialogTitle>

			<DialogContent className="flex flex-col gap-5">
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
