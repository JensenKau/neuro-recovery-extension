import React from "react";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";

const Login = () => {
  return (
    <div className="flex justify-center">
      <div className="flex flex-col gap-4">
        <TextField id="outlined-basic" label="Outlined" variant="outlined" />
        <TextField id="outlined-basic" label="Outlined" variant="outlined" />
        <Button variant="contained">some button</Button>
      </div>
    </div>
  );
};

export default Login;
