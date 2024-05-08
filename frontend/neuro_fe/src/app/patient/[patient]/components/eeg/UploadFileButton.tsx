"use client";

import { Button, Typography } from "@mui/material";
import { UploadFile } from "@mui/icons-material";
import React, { useState } from "react";

interface UplaodFileButtonProps {
	label: string;
	onChange(file: File): void;
}

const UploadFileButton = ({ label, onChange }: UplaodFileButtonProps) => {
	const [filename, setFilename] = useState("");

	const fileChanged = (e: React.ChangeEvent<HTMLInputElement>) => {
		if (e.target.files) {
			setFilename(e.target.files[0].name);
			onChange(e.target.files[0]);
		}
	}

	return (
		<div>
			<Typography className="text-[#01579b]">{label}</Typography>
			<Button
				component="label"
				className="flex justify-center normal-case p-10 rounded-xl border-2 border-dashed border-black border-[#01579b] transition-colors duration-300 hover:bg-slate-100"
			>
				<UploadFile className="m-3 size-10 text-slate-400" />
				<Typography className="my-auto text-slate-400 text-lg">
					{filename.length > 0 ? `Upload ${filename}` : "Upload From Computer"}
				</Typography>
				<input type="file" hidden onChange={fileChanged} />
			</Button>
		</div>
	);
};

export default UploadFileButton;
