"use client";
import React, { ComponentType, useState } from "react";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import CustomButton, { CustomButtonProps } from "./CustomButton";
import Chip from "@mui/material/Chip";

export interface FormProps {
  title: string;
  clicked?: () => void;
  ButtonComponent: ComponentType<CustomButtonProps>;
  buttonProps: CustomButtonProps;
  onSubmit?: (email: string) => void;
  submitFormInfo: string;
}

export default function Form({
  title,
  clicked,
  ButtonComponent,
  buttonProps,
  onSubmit,
  submitFormInfo,
}: FormProps) {
  const [open, setOpen] = React.useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const [email, setEmail] = useState("");
  const [emailList, setEmailList] = useState<string[]>([
    "yteo0019@student.monash.edu",
    "ianteohyx@gmail.com",
  ]);

  const handleAddEmail = () => {
    if (email) {
      setEmailList([...emailList, email]);
      setEmail(""); // Clear the input field after adding
    }
  };

  const handleFormSubmit = (event: React.FormEvent<HTMLFormElement>) => {};

  return (
    <React.Fragment>
      <ButtonComponent {...buttonProps} clicked={handleClickOpen} />
      <Dialog
        open={open}
        onClose={handleClose}
        PaperProps={{
          component: "form",
          onSubmit: handleFormSubmit,
          style: { width: "30%", height: "60%" },
        }}
      >
        <DialogTitle
          sx={{ textAlign: "center", color: "#1976d2", fontSize: "35px" }}
        >
          {title}
        </DialogTitle>
        <DialogContent>
          <div className="flex">
            <TextField
              autoFocus
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              margin="dense"
              id="other-user-email"
              name="userEmail"
              label="Add users to share profile with"
              type="email"
              fullWidth
              variant="outlined"
            />
            <div className="m-[8px]">
              <CustomButton
                clicked={handleAddEmail}
                style="outlined"
                buttonWidth="25px"
                buttonHeight="57px"
              >
                Add
              </CustomButton>
            </div>
          </div>

          <div>
            <div className="text-blue-600 mt-[20px] mb-[10px] ml-[5px]">
              User with access
            </div>
            {emailList.map((userEmail, index) => (
              <Chip
                key={index}
                label={userEmail}
                onDelete={() => {
                  const newEmailList = [...emailList];
                  newEmailList.splice(index, 1);
                  setEmailList(newEmailList);
                }}
                style={{
                  margin: "5px",
                  backgroundColor: "#bae6fd",
                  height: "50px",
                  width: "380px",
                  justifyContent: "space-between",
                }}
              />
            ))}
          </div>
        </DialogContent>
        <DialogActions className="justify-center ml-[20px] mr-[20px] mb-[10px] mt-[10px]">
          <Button
            type="submit"
            onClick={clicked}
            variant="outlined"
            className="w-full"
          >
            {submitFormInfo}
          </Button>
        </DialogActions>
      </Dialog>
    </React.Fragment>
  );
}
