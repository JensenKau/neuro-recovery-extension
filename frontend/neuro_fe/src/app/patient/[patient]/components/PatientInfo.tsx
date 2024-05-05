"use client";

import React from "react";
import PatientInfoItem from "./PatientInfoItem";

const PatientInfo = () => {
	return (
		<div className="flex flex-col gap-5">
			<div className="my-auto text-3xl text-blue-600">Patient Information</div>

			<div className="grid grid-cols-2 mx-3">
				<div className="grid grid-cols-3 gap-3">
					<PatientInfoItem label="First Name" value="Jensen" />
					<PatientInfoItem label="Last Name" value="Jensen" />
					<PatientInfoItem label="Age" value="Jensen" />
					<PatientInfoItem label="Gender" value="Jensen" />
				</div>

				<div className="grid grid-cols-3 gap-3">
					<PatientInfoItem label="ROSC" value="Jensen" />
					<PatientInfoItem label="OHCA" value="Jensen" />
					<PatientInfoItem label="Shockable Rhythm" value="Jensen" />
					<PatientInfoItem label="TTM" value="Jensen" />
				</div>
			</div>
		</div>
	);
};

export default PatientInfo;
