import React from "react";
import Button from "@mui/material/Button";
import Stack from '@mui/material/Stack';
import TextField from "@mui/material/TextField";
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';

const Login = () => {
  return (
    <div className="flex justify-center items-center h-screen bg-sky-100">
      <div className="w-2/5 bg-white rounded-md">
        <Stack spacing={2} className="px-10 py-16 w-full">
          <h1 className="text-center">User Login</h1>
          <TextField label="Email" variant="outlined" />
          <TextField type="password" label="Password" variant="outlined" />
          <div className="grid grid-cols-2">
            <FormControlLabel control={<Checkbox />} label="Remember Me" />
            <h1 className="text-right my-auto">Forgot Password</h1>
          </div>
          <Button className="bg-[#1976d2]" variant="contained" color="primary">Login</Button>
          <h1 className="text-center">New User? Sign up Now</h1>
        </Stack>
      </div>
    </div>
  );
};

export default Login;
