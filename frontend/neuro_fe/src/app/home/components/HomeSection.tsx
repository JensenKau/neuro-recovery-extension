"use client";

import React from "react";
import PatientFolder from "./PatientFolder";
import { ShortPatient } from "../interface";
import { Typography } from "@mui/material";

interface HomeSectionProp {
	patients: ShortPatient[];
	message: string
}

const HomeSection = ({ patients, message }: HomeSectionProp) => {
  return (
		<>
			{
				(patients.length > 0) ?
				<div className="flex flex-wrap gap-4 w-full my-5">
					{patients.map((patient) => <PatientFolder patient={patient} key={patient.id} />)}
				</div> :
				<Typography variant="h5" className="text-center">{message}</Typography>
			}
		</>
	);
};

export default HomeSection;