import React from "react";
import HomeIcon from "../components/HomeIcon";
import WelcomeMessage from "../components/WelcomeMessage";
import MyItems from "../components/MyItems";
import AddIcon from "@mui/icons-material/Add";
import CustomButton from "../components/CustomButton";
import PatientForm from "../components/PatientForm";

const page = () => {
  return (
    <div>
      <HomeIcon />
      <WelcomeMessage header="Welcome back, " name="Ian" />
      <MyItems
        initialItems={["Ian", "Jack"]}
        FormButtonComponent={PatientForm}
        FormButtonProps={{title: "New Patient", ButtonComponent: CustomButton, buttonProps:{children: "Add Patients", icon: AddIcon} }}
      >
        My Patients
      </MyItems>

      <MyItems initialItems={["Ian", "Jack"]}>Shared With Me</MyItems>

    </div>
  );
};

export default page;
