"use client";

import React from "react";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Button from "@mui/material/Button";
import Stack from '@mui/material/Stack';
import TextField from "@mui/material/TextField";
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';
import Link from "next/link";
import { Typography } from "@mui/material";
import { ToastContainer, toast } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberme, setRememberme] = useState(false);
  const router = useRouter();

  const submitLogin = async () => {
    const res = await fetch("http://localhost:3000/api/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        rememberme: rememberme,
        email: email,
        password: password
      })
    });

    if (res.status === 200) {
      router.push("/home");
    } else {
      toast.error("Login Failed", {
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

  return (
    <div className="flex justify-center items-center h-screen bg-sky-100">
      <div className="w-2/5 bg-white rounded-md">
        <Stack spacing={2} className="px-10 py-10 w-full">
          <Typography variant="h5" className="text-center text-2xl">
            User Login
          </Typography>

          <TextField label="Email" variant="outlined" onChange={(e) => setEmail(e.target.value)} />
          <TextField type="password" label="Password" variant="outlined" onChange={(e) => setPassword(e.target.value)} />

          <div className="grid grid-cols-2">
            <FormControlLabel control={<Checkbox onChange={(e) => setRememberme(e.target.checked)} />} label="Remember Me" />
          </div>

          <Button variant="contained" onClick={submitLogin}>Login</Button>
          <h1 className="text-center">
            New User? {" "}
            <Link href="/register">
              <u>Sign up Now</u>
            </Link>
          </h1>
        </Stack>
      </div>
    </div>
  );
};

export default Login;
