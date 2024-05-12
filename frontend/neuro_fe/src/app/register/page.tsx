"use client";

import React from "react";
import { useState } from "react";
import Button from "@mui/material/Button";
import Stack from '@mui/material/Stack';
import { TextField } from "@mui/material";
import { useRouter } from "next/navigation";
import { ToastContainer, toast } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';

const Register = () => {
	const router = useRouter();

	const [lastName, setLastName] = useState("");
	const [firstName, setFirstName] = useState("");
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");
	const [confirmPassword, setConfirmPassword] = useState("");

	const submitSignUp = async () => {
		if (password === confirmPassword) {
			const res = await fetch("http://localhost:3000/api/signup", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					firstname: firstName,
					lastname: lastName,
					email: email,
					password: password
				})
			});

			if (res.status === 200) {
				router.push("/login");
			} else {
				toast.error("Registration Failed", {
					position: "top-center",
					autoClose: 5000,
					hideProgressBar: false,
					closeOnClick: true,
					pauseOnHover: true,
					draggable: true,
					progress: undefined,
					theme: "light",
				});
			}
		}
	};

	return (
		<div className="flex justify-center items-center h-screen bg-sky-100">
			<div className="w-2/5 bg-white rounded-md">
				<Stack spacing={2} className="px-10 py-16 w-full">
					<h1 className="text-center">User Sign Up</h1>

					<TextField label="Last Name" onChange={(e) => setLastName(e.target.value)} />
					<TextField label="First Name" onChange={(e) => setFirstName(e.target.value)} />
					<TextField label="Email" onChange={(e) => setEmail(e.target.value)} />
					<TextField label="Password" type="password" onChange={(e) => setPassword(e.target.value)} />
					<TextField label="Re-Type Password" type="password" onChange={(e) => setConfirmPassword(e.target.value)} />

					<Button variant="contained" onClick={submitSignUp}>Sign Up</Button>
				</Stack>
			</div>
		</div>
	);
};

export default Register;