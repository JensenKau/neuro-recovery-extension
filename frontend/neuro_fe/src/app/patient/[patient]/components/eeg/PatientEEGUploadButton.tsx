"use client";

import { Button } from "@mui/material";
import { FileUpload } from "@mui/icons-material";
import React, { useState } from "react";
import PatientEEGUploadForm from "./PatientEEGUploadForm";
import { Patient } from "@/app/interface";

interface PatientEEGUploadButtonProps {
	className?: string;
	patient: Patient | null;
	onUpload(): void;
}

const PatientEEGUploadButton = ({ className = "", patient, onUpload }: PatientEEGUploadButtonProps) => {
	const [open, setOpen] = useState(false);

	return (
		<>
			<Button variant="contained" className={`${className} flex gap-2`} onClick={() => setOpen(true)}>
				<FileUpload />
				Upload EEG
			</Button>
			<PatientEEGUploadForm patient={patient} open={open} onClose={setOpen} onUpload={onUpload} />
		</>
	);
};

export default PatientEEGUploadButton;