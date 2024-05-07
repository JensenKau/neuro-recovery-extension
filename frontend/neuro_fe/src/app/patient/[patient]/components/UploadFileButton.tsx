"use client";

import { Button, Typography } from "@mui/material";
import { UploadFile } from "@mui/icons-material";
import React from "react";

interface UplaodFileButtonProps {
	label: string;
	onChange(file: File): void;
}

const UploadFileButton = ({ label, onChange }: UplaodFileButtonProps) => {
	return (
		<div>
			<Typography className="text-[#01579b]">{label}</Typography>
			<Button
				component="label"
				className="flex justify-center normal-case p-14 rounded-xl border-2 border-dashed border-black border-[#01579b] transition-colors duration-300 hover:bg-slate-100"
			>
				<UploadFile className="m-3 size-10 text-slate-400" />
				<Typography className="my-auto text-slate-400 text-lg">
					Upload from Computer
				</Typography>
				<input type="file" hidden onChange={(e) => {e.target.files ? onChange(e.target.files[0]) : ""}} />
			</Button>
		</div>
	);
};

export default UploadFileButton;
