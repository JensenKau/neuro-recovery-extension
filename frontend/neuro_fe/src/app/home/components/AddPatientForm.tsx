"use client";

import React from "react";
import { useState } from "react";
import {
	TextField,
	Button,
	Select,
	MenuItem,
	InputLabel,
	FormControl,
} from "@mui/material";

const AddPatientForm = () => {
	const [lastname, setLastname] = useState("");
	const [firstname, setFirstname] = useState("");
	const [age, setAge] = useState<number | null>(null);
	const [gender, setGender] = useState<string | null>("");
	const [rosc, setRosc] = useState<number | null>(null);
	const [ohca, setOhca] = useState<boolean | null>(null);
	const [sr, setSr] = useState<boolean | null>(null);
	const [ttm, setTtm] = useState<number | null>(null);



	return (
		<div className="flex flex-col bg-red-500">
			<TextField required label="Last Name" variant="outlined" />
			<TextField required label="First Name" variant="outlined" />
			<TextField label="Age" type="number" variant="outlined" />

			<FormControl>
				<InputLabel>Gender</InputLabel>
				<Select value={gender} label="Gender" onChange={() => {}}>
					<MenuItem value={"Male"}>Male</MenuItem>
					<MenuItem value={"Female"}>Female</MenuItem>
					<MenuItem value={""}>None</MenuItem>
				</Select>
			</FormControl>

			<TextField label="ROSC" type="number" variant="outlined" />

			<FormControl>
				<InputLabel>OHCA</InputLabel>
				<Select value={ohca} label="ohca">
					<MenuItem value={"Yes"}>Yes</MenuItem>
					<MenuItem value={"No"}>No</MenuItem>
					<MenuItem value={""}>None</MenuItem>
				</Select>
			</FormControl>

			<FormControl>
				<InputLabel>Shockable Rhythm</InputLabel>
				<Select value={sr} label="Shockable Rhythm">
					<MenuItem value={"Yes"}>Yes</MenuItem>
					<MenuItem value={"No"}>No</MenuItem>
					<MenuItem value={""}>None</MenuItem>
				</Select>
			</FormControl>

			<TextField label="TTM" type="number" variant="outlined" />
			<Button>what is this</Button>
		</div>
	);
};

export default AddPatientForm;
