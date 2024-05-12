"use client";

import React from "react";
import { useState } from "react";
import { Button } from "@mui/material";
import Share from "@mui/icons-material/Share";
import ShareProfileForm from "./ShareProfileForm";
import { Patient } from "@/app/interface";

interface ShareProfileButtonProps {
	className?: string;
	patient: Patient | null;
	onUpdate(value: boolean): void;
}

const ShareProfileButton = ({ className = "", patient, onUpdate }: ShareProfileButtonProps) => {
	const [open, setOpen] = useState(false);

	return (
		<>
			<Button className={`${className} flex gap-2`} onClick={() => setOpen(true)}>
				<Share />
				Share Profile
			</Button>
			<ShareProfileForm open={open} onClose={setOpen} patient={patient} onUpdate={onUpdate} />
		</>
	);
};

export default ShareProfileButton;