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
	const [heaUrl, setHeaUrl] = useState("");
	const [matUrl, setMatUrl] = useState("");

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
				<UploadFileButton label=".hea" onChange={setHeaUrl} />
				<UploadFileButton label=".mat" onChange={setMatUrl} />

				<Button variant="contained" className="w-full">
					Upload
				</Button>
			</DialogContent>
		</Dialog>
	);
};

export default PatientEEGUploadForm;
