"use client";

import { Button } from "@mui/material";
import { FileUpload } from "@mui/icons-material";
import React, { useState } from "react";
import PatientEEGUploadForm from "./PatientEEGUploadForm";

interface PatientEEGUploadButtonProps {
	className?: string;
}

const PatientEEGUploadButton = ({className = ""}: PatientEEGUploadButtonProps) => {
	const [open, setOpen] = useState(false);

	return (
		<>
			<Button variant="contained" className={`${className} flex gap-2`} onClick={() => setOpen(true)}>
				<FileUpload />
				Upload EEG
			</Button>
			<PatientEEGUploadForm open={open} onClose={setOpen} />
		</>
	);
};

export default PatientEEGUploadButton;