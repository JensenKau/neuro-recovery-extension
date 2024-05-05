"use client";

import React from "react";
import { Button } from "@mui/material";
import FolderIcon from "@mui/icons-material/Folder";
import { ShortPatient } from "../interface";
import { Typography } from "@mui/material";
import Link from "next/link";

interface PatientFolderProps {
	patient: ShortPatient;
}

const PatientFolder = ({ patient }: PatientFolderProps) => {
	return (
		<>
			{patient && (
				<Link href={`/patient/${patient.id}`}>
					<Button className="capitalize w-[250px] h-[60px] bg-[#b3e5fc] rounded-xl hover:bg-[#64b5f6]">
						<div className="grid grid-cols-4 w-full h-full items-center">
							<FolderIcon className="text-[#0277bd] text-3xl col-span-1 w-full" />
							<Typography
								variant="h5"
								className="text-black text-lg text-base col-span-3 text-left m-3"
							>
								{patient.name}
							</Typography>
						</div>
					</Button>
				</Link>
			)}
		</>
	);
};

export default PatientFolder;
