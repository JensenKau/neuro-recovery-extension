import React from "react";
import MyItems from "../components/MyItems";
import AddIcon from "@mui/icons-material/Add";
import CustomButton from "../components/CustomButton";
import PatientForm from "../components/PatientForm";
import Folder from "@mui/icons-material/Folder";

const page = () => {
  return (
    <div>
       <div className="mb-[80px] mt-[50px] ml-[30px] text-5xl">
          Welcome Back, <span className="text-blue-600">Doctor</span>
        </div>
      <MyItems
        initialItems={["Ian", "Jack"]}
        FormButtonComponent={PatientForm}
        FormButtonProps={{
          title: "New Patient",
          ButtonComponent: CustomButton,
          buttonProps: { children: "Add Patients", icon: AddIcon },
          submitFormInfo: "add patient"
        }}
        chipsIcon={Folder}
        chipsHeight="55px"
        chipsWidth="250px"
        chipsClickable
      >
        My Patients
      </MyItems>

      <MyItems
        initialItems={["Ian", "Jack"]}
        chipsIcon={Folder}
        chipsHeight="55px"
        chipsWidth="250px"
        chipsClickable
      >
        Shared With Me
      </MyItems>
    </div>
  );
};

export default page;
