"use client";

import React from "react";
import { useState } from "react";
import { Button } from "@mui/material";
import Share from "@mui/icons-material/Share";
import ShareProfileForm from "./ShareProfileForm";

interface ShareProfileButtonProps {
	className?: string;
}

const ShareProfileButton = ({ className = "" }: ShareProfileButtonProps) => {
	const [open, setOpen] = useState(false);

	return (
		<>
			<Button className={`${className} flex gap-2`} onClick={() => setOpen(true)}>
				<Share />
				Share Profile
			</Button>
			<ShareProfileForm open={open} onClose={setOpen} />
		</>
	);
};

export default ShareProfileButton;