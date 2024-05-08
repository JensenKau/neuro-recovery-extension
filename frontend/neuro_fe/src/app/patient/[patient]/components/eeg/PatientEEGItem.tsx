"use client";

import { Button, Typography } from "@mui/material";
import { DescriptionOutlined } from "@mui/icons-material";
import React from "react";
import Link from "next/link";

interface PatientEEGItemProps {
	patient: number;
	label: string;
	datetime: string;
}

const PatientEEGItem = ({ patient, label, datetime }: PatientEEGItemProps) => {
	return (
		<Link href={`/patient/${patient}/${label}`}>
			<Button className="flex justify-between w-full py-3 px-5 normal-case rounded-xl bg-[#b3e5fc] hover:bg-[#64b5f6] text-black">
				<div className="flex gap-3">
					<DescriptionOutlined className="size-10" />
					<Typography variant="h5" className="my-auto text-lg">{label}</Typography>
				</div>
				<Typography variant="h5" className="my-auto text-lg">{datetime}</Typography>
			</Button>
		</Link>
	);
};

export default PatientEEGItem;