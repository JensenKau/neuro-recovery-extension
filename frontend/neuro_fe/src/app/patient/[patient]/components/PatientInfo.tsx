"use client";

import React, { useEffect, useState } from "react";
import PatientInfoItem from "./PatientInfoItem";
import { Patient } from "@/app/interface";
import { Typography } from "@mui/material";

interface PatientInfoProps {
	patient: Patient | null;
}

const PatientInfo = ({ patient }: PatientInfoProps) => {
	return (
		<div className="flex flex-col gap-5">
			<div className="my-auto text-3xl text-blue-600">Patient Information</div>

			{patient ? (
				<div className="grid grid-cols-2 mx-3">
					<div className="grid grid-cols-3 gap-3">
						<PatientInfoItem label="First Name" value={`${patient.first_name}`} />
						<PatientInfoItem label="Last Name" value={`${patient.last_name}`} />
						<PatientInfoItem label="Age" value={`${patient.age}`} />
						<PatientInfoItem label="Gender" value={`${patient.sex}`} />
					</div>

					<div className="grid grid-cols-3 gap-3">
						<PatientInfoItem label="ROSC" value={`${patient.rosc}`} />
						<PatientInfoItem label="OHCA" value={`${patient.ohca}`} />
						<PatientInfoItem label="Shockable Rhythm" value={`${patient.shockable_rhythm}`} />
						<PatientInfoItem label="TTM" value={`${patient.ttm}`} />
					</div>
				</div>) : (<Typography>Error</Typography>)}


		</div>
	);
};

export default PatientInfo;
