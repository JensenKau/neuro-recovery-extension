"use client";

import React from "react";
import { useState } from "react";
import AddPatientForm from "./AddPatientForm";
import { ShortPatient } from "../interface";
import { Button } from "@mui/material";

interface AddPatientButtonProps {
	onSubmit(value: ShortPatient): void;
	className?: string;
}

const AddPatientButton = ({ onSubmit, className = "" }: AddPatientButtonProps) => {
	const [open, setOpen] = useState(false);

	return (
		<>
			<Button className={`${className}`}>Add Patient</Button>
			<AddPatientForm open={open} onClose={setOpen} onSubmit={onSubmit} />
		</>
	);
};

export default AddPatientButton;
