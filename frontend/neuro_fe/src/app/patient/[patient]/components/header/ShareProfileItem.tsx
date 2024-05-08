"use client";

import { Typography } from "@mui/material";
import Delete from "@mui/icons-material/Delete";
import React from "react";

interface ShareProfileItemProps {
	email: string;
	deletable?: boolean;
	onDelete?(email: string): void;
}

const ShareProfileItem = ({
	email,
	deletable = false,
	onDelete = () => {},
}: ShareProfileItemProps) => {
	return (
		<div className="flex justify-between bg-[#b3e5fc] rounded-xl py-3 px-5">
			<Typography variant="h5" className="text-base my-auto">
				{email}
			</Typography>
			<div
				className={`hover:bg-[#64b5f6] rounded-full p-1 ${deletable ? "opacity-100" : "opacity-0"}`}
				onClick={() => onDelete(email)}
			>
				<Delete className="text-[#d32f2f] font-2xl" />
			</div>
		</div>
	);
};

export default ShareProfileItem;
