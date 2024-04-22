"use client"
import * as React from 'react';
import Button from '@mui/material/Button';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';

interface Props{
    children: string;
    buttonInfo: string;
}

export default function SimpleSnackbar({children, buttonInfo}: Props) {
  const [open, setOpen] = React.useState(false);

  const handleClick = () => {
    setOpen(true);
  };

  const handleClose = (event: React.SyntheticEvent | Event, reason?: string) => {
    if (reason === 'clickaway') {
      return;
    }
    setOpen(false);
  };

  return (
    <div className="ml-[10px]">
      <Button onClick={handleClick} variant="outlined">{buttonInfo}</Button>
      <Snackbar open={open} autoHideDuration={6000} onClose={handleClose} anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}>
      <Alert
          onClose={handleClose}
          severity="success"
          variant="filled"
          sx={{ width: '100%' }}
        >
          {children}
        </Alert>
      </Snackbar>
    </div>
  );
}