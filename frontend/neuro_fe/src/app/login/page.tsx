"use client";

import React from "react";
import { useState } from "react";
import Button from "@mui/material/Button";
import Stack from '@mui/material/Stack';
import TextField from "@mui/material/TextField";
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';

import { apiSignIn } from "../utils/api";

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberme, setRememberme] = useState(false);

  const submitLogin = async () => {
    await fetch(
      "http://localhost:3000/api/login",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
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
          <h1 className="text-center">User Login</h1>
          <TextField label="Email" variant="outlined" onChange={(e) => setEmail(e.target.value)} />
          <TextField type="password" label="Password" variant="outlined" onChange={(e) => setPassword(e.target.value)} />
          <div className="grid grid-cols-2">
            <FormControlLabel control={<Checkbox onChange={(e) => setRememberme(e.target.checked)} />} label="Remember Me" />
            <h1 className="text-right my-auto"><u>Forgot Password</u></h1>
          </div>
          <Button className="bg-[#1976d2]" variant="contained" color="primary" onClick={submitLogin}>Login</Button>
          <h1 className="text-center">New User? <u>Sign up Now</u></h1>
        </Stack>
      </div>
    </div>
  );
};

export default Login;
