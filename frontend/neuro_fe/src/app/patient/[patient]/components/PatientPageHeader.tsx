"use client";

import React from "react";
import { Divider } from "@mui/material";
import HomeIcon from "./HomeIcon";
import ShareProfileButton from "./ShareProfileButton";
import { Patient } from "@/app/interface";

interface PatientPageHeaderProps {
	patient: Patient | null;
}

const PatientPageHeader = ({patient}: PatientPageHeaderProps) => {
	return (
		<div>
			<div className="flex justify-between">
				<div className="flex gap-3">
					<HomeIcon />
					<div className="my-auto text-4xl">
						My Patients {">"} <span className="text-blue-600">{`${patient ? patient.name : ""}`}</span>
					</div>
				</div>
				<ShareProfileButton patient={patient} className="my-auto h-3/4 p-3" />
			</div>
			<Divider className="my-3" />
		</div>
	);
};

export default PatientPageHeader;
