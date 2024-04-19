import React from "react";
import TextField from "@mui/material/TextField";

interface SignupTextFieldProps {
	error: boolean,
	text: string,
	func(s: string): void,
	password?: boolean
}

const SignupTextField = ({ error, text, func, password = false }: SignupTextFieldProps) => {
	return (
		<>
			{
				(error) ?
					<TextField error label={text} variant="outlined" type={(password) ? "password" : undefined} onChange={(e) => func(e.target.value)} /> :
					<TextField label={text} variant="outlined" type={(password) ? "password" : undefined} onChange={(e) => func(e.target.value)} />
			}
		</>
	);
};

export default SignupTextField;