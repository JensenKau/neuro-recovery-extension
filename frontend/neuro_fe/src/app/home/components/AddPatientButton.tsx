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
			<Button variant="contained" className={`${className}`} onClick={() => setOpen(true)}>Add Patient</Button>
			<AddPatientForm open={open} onClose={setOpen} onSubmit={onSubmit} />
		</>
	);
};

export default AddPatientButton;
