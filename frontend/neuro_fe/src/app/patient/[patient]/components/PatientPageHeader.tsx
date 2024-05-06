"use client";

import React from "react";
import { Divider } from "@mui/material";
import HomeIcon from "./HomeIcon";
import ShareProfileButton from "./ShareProfileButton";

const PatientPageHeader = () => {
	return (
		<div>
			<div className="flex justify-between">
				<div className="flex gap-3">
					<HomeIcon />
					<div className="my-auto text-4xl">
						My Patients {">"} <span className="text-blue-600">klsdjfds</span>
					</div>
				</div>
				<ShareProfileButton className="my-auto h-3/4 p-3" />
			</div>
			<Divider className="my-3" />
		</div>
	);
};

export default PatientPageHeader;
