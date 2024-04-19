"use client";

import React from "react";
import { useState } from "react";
import Button from "@mui/material/Button";
import Stack from '@mui/material/Stack';
import Alert from "@mui/material/Alert";
import SignupTextField from "./components/SignupTextField";

import { apiSignUp } from "../utils/api";

const Register = () => {
	const [lastName, setLastName] = useState("");
	const [firstName, setFirstName] = useState("");
	const [email, setEmail] = useState("");
	const [password, setPassword] = useState("");
	const [confirmPassword, setConfirmPassword] = useState("");

	const [previousEmail, setPreviousEmail] = useState("");
	const [submitAttempt, setSubmitAttempt] = useState(false);

	const blankSubmitCheck = !submitAttempt || (lastName.length > 0 && firstName.length > 0 && email.length > 0 && password.length > 0 && confirmPassword.length > 0);
	const emailTakenCheck = previousEmail.length === 0 || email !== previousEmail;
	const confirmPasswordCheck = confirmPassword.length === 0 || password === confirmPassword;

	const getErrorMsg = () => {
		if (!blankSubmitCheck) { return "Fields Cannot be Blank" }
		if (!emailTakenCheck) { return "Email has been Taken" }
		if (!confirmPasswordCheck) { return "Password and Retype Password Must be the Same" }

		return "";
	}

	const submitSignUp = async () => {
		await fetch(
			"http://localhost:3000/api/signup",
			{
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
			}
		);
	}

	return (
		<div className="flex justify-center items-center h-screen bg-sky-100">
			<div className="w-2/5 bg-white rounded-md">
				<Stack spacing={2} className="px-10 py-16 w-full">
					<h1 className="text-center">User Sign Up</h1>
					{getErrorMsg() !== "" && <Alert severity="error">{getErrorMsg()}</Alert>}

					<SignupTextField error={!blankSubmitCheck && lastName.length === 0} text="Last Name" func={setLastName} />
					<SignupTextField error={!blankSubmitCheck && firstName.length === 0} text="First Name" func={setFirstName} />
					<SignupTextField error={(!blankSubmitCheck && email.length === 0) || (!emailTakenCheck && email === previousEmail)} text="Email" func={setEmail} />
					<SignupTextField error={!blankSubmitCheck && password.length === 0} text="Password" password={true} func={setPassword} />
					<SignupTextField error={(!blankSubmitCheck && confirmPassword.length === 0) || !confirmPasswordCheck} text="Retype Password" password={true} func={setConfirmPassword} />

					<Button disabled={getErrorMsg() !== ""} className="bg-[#1976d2]" variant="contained" color="primary" onClick={submitSignUp}>Sign Up</Button>
				</Stack>
			</div>
		</div>
	);
};

export default Register;