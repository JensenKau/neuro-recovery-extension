"use client";

import React, { useEffect, useState } from "react";
import PatientEEGItem from "./PatientEEGItem";

const PatientEEG = () => {
	const [time, setTime] = useState("");

	useEffect(() => {
		setTime(new Date().toLocaleString());
	}, [])

	return (
		<div className="flex flex-col gap-5">
			<div className="my-auto text-3xl text-blue-600">Patient EEGs</div>
			<PatientEEGItem label="slkdfj" datetime={time}/>
		</div>
	);
};

export default PatientEEG;
