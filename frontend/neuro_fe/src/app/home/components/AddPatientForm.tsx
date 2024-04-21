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
	Dialog,
	DialogTitle,
	DialogContent,
	DialogActions,
} from "@mui/material";
import { ShortPatient } from "../interface";

interface AddPatientFormProps {
	open: boolean;
	onClose(value: boolean): void;
	onSubmit(value: ShortPatient): void;
}

const AddPatientForm = ({ open, onClose, onSubmit }: AddPatientFormProps) => {
	const [lastname, setLastname] = useState("");
	const [firstname, setFirstname] = useState("");
	const [age, setAge] = useState<number | null>(null);
	const [gender, setGender] = useState<string | null>(null);
	const [rosc, setRosc] = useState<number | null>(null);
	const [ohca, setOhca] = useState<boolean | null>(null);
	const [sr, setSr] = useState<boolean | null>(null);
	const [ttm, setTtm] = useState<number | null>(null);

	const createPatient = async (): Promise<ShortPatient> => {
		const res = await fetch(
			"http://localhost:3000/api/patient/create_patient",
			{
				method: "POST",
				credentials: "include",
				headers: {
					"Content-Type": "application/json"
				},
				body: JSON.stringify({
					lastname: lastname,
					firstname: firstname,
					age: age,
					gender: gender,
					rosc: rosc,
					ohca: ohca,
					sr: sr,
					ttm: ttm
				})
			}
		);

		return await res.json();
	};

	return (
		<Dialog
			open={open}
			onClose={() => onClose(false)}
			PaperProps={{
				component: "form",
				onSubmit: (event: React.FormEvent<HTMLFormElement>) => {
					event.preventDefault();
					createPatient().then((response) => onSubmit(response));
					onClose(false);
				},
			}}
		>
			<DialogTitle>Add a Patient</DialogTitle>
			<DialogContent className="flex flex-col gap-3 w-full m-3 p-3">
				<TextField
					required
					label="Last Name"
					variant="outlined"
					onChange={(e) => setLastname(e.target.value)}
				/>
				<TextField
					required
					label="First Name"
					variant="outlined"
					onChange={(e) => setFirstname(e.target.value)}
				/>

				<TextField
					label="Age"
					type="number"
					variant="outlined"
					onChange={(e) => {
						const value = e.target.value;
						setAge(value === "" ? null : parseInt(value));
					}}
				/>

				<FormControl>
					<InputLabel>Gender</InputLabel>
					<Select
						value={gender === null ? "" : gender}
						label="Gender"
						onChange={(e) => {
							const value = e.target.value;
							setGender(value === "" ? null : value);
						}}
					>
						<MenuItem value={""}>None</MenuItem>
						<MenuItem value={"Male"}>Male</MenuItem>
						<MenuItem value={"Female"}>Female</MenuItem>
					</Select>
				</FormControl>

				<TextField
					label="ROSC"
					type="number"
					variant="outlined"
					onChange={(e) => {
						const value = e.target.value;
						setRosc(value === "" ? null : parseFloat(value));
					}}
				/>

				<div className="flex flex-row gap-3 w-full">
					<FormControl className="w-full">
						<InputLabel>OHCA</InputLabel>
						<Select
							value={ohca === null ? "" : ohca ? "Yes" : "No"}
							label="ohca"
							onChange={(e) => {
								const value = e.target.value;
								setOhca(value === "" ? null : value === "Yes" ? true : false);
							}}
						>
							<MenuItem value={""}>None</MenuItem>
							<MenuItem value={"Yes"}>Yes</MenuItem>
							<MenuItem value={"No"}>No</MenuItem>
						</Select>
					</FormControl>

					<FormControl className="w-full">
						<InputLabel>Shockable Rhythm</InputLabel>
						<Select
							value={sr === null ? "" : sr ? "Yes" : "No"}
							label="Shockable Rhythm"
							onChange={(e) => {
								const value = e.target.value;
								setSr(value === "" ? null : value === "Yes" ? true : false);
							}}
						>
							<MenuItem value={""}>None</MenuItem>
							<MenuItem value={"Yes"}>Yes</MenuItem>
							<MenuItem value={"No"}>No</MenuItem>
						</Select>
					</FormControl>
				</div>

				<TextField
					label="TTM"
					type="number"
					variant="outlined"
					onChange={(e) => {
						const value = e.target.value;
						setTtm(value === "" ? null : parseInt(value));
					}}
				/>
			</DialogContent>
			<DialogActions>
				<Button type="submit">Add Patient</Button>
			</DialogActions>
		</Dialog>
	);
};

export default AddPatientForm;
