"use client";

import React from "react";
import { Divider } from "@mui/material";
import HomeIcon from "./HomeIcon";

const PatientPageHeader = () => {
	return (
		<div>
			<div className="flex gap-3">
				<HomeIcon />
				<div className="my-auto text-4xl">
					My Patients {">"} <span className="text-blue-600">klsdjfds</span>
				</div>
			</div>
			<Divider className="my-3" />
		</div>
	);
};

export default PatientPageHeader;
