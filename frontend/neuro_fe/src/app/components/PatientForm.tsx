"use client";
import React, { ComponentType } from "react";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import { CustomButtonProps } from "./CustomButton";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select, { SelectChangeEvent } from "@mui/material/Select";
import Box from "@mui/material/Box";

export interface FormProps {
  title: string;
  clicked?: () => void;
  ButtonComponent: ComponentType<CustomButtonProps>;
  buttonProps: CustomButtonProps;
  onSubmit?: (email: string) => void;
}

export default function Form({
  title,
  clicked,
  ButtonComponent,
  buttonProps,
  onSubmit,
}: FormProps) {
  const [open, setOpen] = React.useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const handleFormSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const formJson = Object.fromEntries(formData.entries());
    const firstName = formJson.firstName as string;
    const lastName = formJson.lastName as string;
    onSubmit === undefined ? undefined : onSubmit(firstName + " " + lastName);
    handleClose();
  };

  const [gender, setGender] = React.useState("");

  const handleGenderChange = (event: SelectChangeEvent) => {
    setGender(event.target.value as string);
  };

  const [ohca, setOhca] = React.useState("");

  const handleOhcaChange = (event: SelectChangeEvent) => {
    setOhca(event.target.value as string);
  };

  const [sr, setSr] = React.useState("");

  const handleSrChange = (event: SelectChangeEvent) => {
    setSr(event.target.value as string);
  };

  return (
    <React.Fragment>
      <ButtonComponent {...buttonProps} clicked={handleClickOpen} />
      <Dialog
        open={open}
        onClose={handleClose}
        PaperProps={{
          component: "form",
          onSubmit: handleFormSubmit,
        }}
      >
        <DialogTitle sx={{ textAlign: "center" }}>{title}</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            required
            margin="dense"
            id="first name"
            name="firstName"
            label="First Name"
            fullWidth
            variant="outlined"
          />
          <TextField
            autoFocus
            required
            margin="dense"
            id="last name"
            name="lastName"
            label="Last Name"
            fullWidth
            variant="outlined"
          />
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
            }}
          >
            <TextField
              autoFocus
              required
              margin="dense"
              id="age"
              name="age"
              label="Age"
              variant="outlined"
              type="number"
              inputProps={{ step: 1, min: 0 }}
              sx={{ flex: 1, marginRight: "10px" }}
            />
            <FormControl sx={{ flex: 1, marginLeft: "10px", marginTop: "8px" }}>
              <InputLabel id="gender-drop-down">Gender</InputLabel>
              <Select
                labelId="gender-label-id"
                id="gender-id"
                value={gender}
                label="Gender"
                onChange={handleGenderChange}
                fullWidth
              >
                <MenuItem value={"Male"}>Male</MenuItem>
                <MenuItem value={"Female"}>Female</MenuItem>
                <MenuItem value={"Prefer Not To Say"}>
                  Prefer Not To Say
                </MenuItem>
              </Select>
            </FormControl>
          </div>
          <TextField
            autoFocus
            required
            margin="dense"
            id="rosc"
            name="rosc"
            label="ROSC"
            variant="outlined"
            type="number"
            inputProps={{ step: 0.01, min: 0 }}
            fullWidth
          />
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
            }}
          >
            <FormControl sx={{ flex: 1, marginTop: "8px" }}>
              <InputLabel id="ohca">OHCA</InputLabel>
              <Select
                labelId="ohca-label-id"
                id="ohca-id"
                value={ohca}
                label="ohca"
                onChange={handleOhcaChange}
                fullWidth
              >
                <MenuItem value={"Yes"}>Yes</MenuItem>
                <MenuItem value={"No"}>No</MenuItem>
              </Select>
            </FormControl>

            <FormControl sx={{ flex: 1, marginLeft: "10px", marginTop: "8px" }}>
              <InputLabel id="shockable rhythm">Shockable Rhythm</InputLabel>
              <Select
                labelId="sr-lable-id"
                id="sr-id"
                value={sr}
                label="Shockable Rhythm"
                onChange={handleSrChange}
                fullWidth
              >
                <MenuItem value={"Yes"}>Yes</MenuItem>
                <MenuItem value={"No"}>No</MenuItem>
              </Select>
            </FormControl>
          </div>

          <TextField
            autoFocus
            required
            margin="dense"
            id="ttm"
            name="ttm"
            label="TTM"
            variant="outlined"
            type="number"
            inputProps={{ step: 1, min: 0 }}
            sx={{ flex: 1, marginRight: "10px" }}
            fullWidth
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose}>Cancel</Button>
          <Button type="submit" onClick={clicked}>
            Add Patient
          </Button>
        </DialogActions>
      </Dialog>
    </React.Fragment>
  );
}
