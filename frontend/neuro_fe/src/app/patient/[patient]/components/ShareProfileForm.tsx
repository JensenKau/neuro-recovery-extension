"use client";

import {
	Dialog,
	DialogTitle,
	Typography,
	TextField,
	DialogContent,
	Button,
} from "@mui/material";
import { Add } from "@mui/icons-material";
import React, { useState } from "react";
import ShareProfileItem from "./ShareProfileItem";
import { Patient } from "@/app/interface";

interface ShareProfileForm {
	open: boolean;
	onClose(value: boolean): void;
	patient: Patient | null;
	modifyAccess(newAccess: Array<string>): void;
}

const ShareProfileForm = ({ open, onClose, patient, modifyAccess }: ShareProfileForm) => {
	const [email, setEmail] = useState("");

	const addUser = async () => {
		const res = await fetch("http://localhost:3000/api/patient/add_user", {
			method: "POST",
			credentials: "include",
			body: JSON.stringify({
				patient_id: patient?.id,
				email: email
			})
		});

		if (patient) {
			modifyAccess([...patient.access, email]);
		}

		return await res.json();
	};

	const deleteUser = async (del_email: string) => {
		const res = await fetch("http://localhost:3000/api/patient/delete_user", {
			method: "POST",
			credentials: "include",
			body: JSON.stringify({
				patient_id: patient?.id,
				email: del_email
			})
		});

		if (patient) {
			modifyAccess(patient.access.filter((val) => val !== del_email))
		}

		return await res.json();
	};

	return (
		<Dialog
			open={open}
			onClose={() => onClose(false)}
			fullWidth={true}
			maxWidth="sm"
		>
			<DialogTitle className="mx-auto mt-5 mb-3">
				<div>
					<Typography variant="h5" className="text-[#01579b]">
						New Patient
					</Typography>
				</div>
			</DialogTitle>

			<DialogContent className="flex flex-col gap-4 w-full">
				<div className="flex gap-3 my-2">
					<TextField className="w-full" label="Share with..." onChange={(e) => setEmail(e.target.value)} />
					<Button variant="contained" className="h-auto w-fit flex gap-1" onClick={addUser}>
						<Add className="text-bold" />
						Add
					</Button>
				</div>

				<div className="flex flex-col gap-1">
					<Typography className="text-blue-600 text-xl">Owner</Typography>
					{patient && <ShareProfileItem email={patient.owner} />}
				</div>

				<div className="flex flex-col gap-1">
					<Typography className="text-blue-600 text-xl">
						User with Access
					</Typography>
					<div className="flex flex-col gap-2 overflow-y-scroll overflow-x-hidden h-[200px]">
						{patient && patient.access.map((val) => <ShareProfileItem key={val} email={val} deletable onDelete={deleteUser} />)}
					</div>
				</div>

				<Button variant="contained" onClick={() => onClose(false)}>Done</Button>
			</DialogContent>
		</Dialog>
	);
};

export default ShareProfileForm;
