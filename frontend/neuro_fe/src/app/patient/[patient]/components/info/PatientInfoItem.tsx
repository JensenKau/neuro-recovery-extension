"use client";

import React from "react";
import { Typography } from "@mui/material";

interface PatientInfoItemProps {
	label: string;
	value: string;
}

const PatientInfoItem = ({ label, value }: PatientInfoItemProps) => {
	return (
		<>
			<Typography variant="h5" className="text-lg col-span-1 font-bold">{label}</Typography>
			<Typography variant="h5" className="text-lg col-span-2">{value}</Typography>
		</>
	);
};

export default PatientInfoItem;