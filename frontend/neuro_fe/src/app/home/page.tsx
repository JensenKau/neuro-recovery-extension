import React from "react";
import MyItems from "../components/MyItems";
import AddIcon from "@mui/icons-material/Add";
import CustomButton from "../components/CustomButton";
import PatientForm from "../components/PatientForm";
import Folder from "@mui/icons-material/Folder";

const page = () => {
  return (
    <div className="mt-[60px] mb-[50px] ml-[50px] mr-[50px]">
      <div className="mb-[80px] text-5xl">
        Welcome Back, <span className="text-blue-600">Doctor</span>
      </div>
      <MyItems
        initialItems={["Ian", "Jack"]}
        childrenSize="3xl"
        FormButtonComponent={PatientForm}
        FormButtonProps={{
          title: "New Patient",
          ButtonComponent: CustomButton,
          buttonProps: {
            children: "Add Patients",
            icon: AddIcon,
            style: "outlined",
          },
          submitFormInfo: "add patient",
        }}
        chipsIcon={Folder}
        chipsHeight="55px"
        chipsWidth="250px"
        chipsClickable
        chipsLinkable
      >
        My Patients
      </MyItems>

      <MyItems
        initialItems={["Ian", "Jack"]}
        childrenSize="3xl"
        chipsIcon={Folder}
        chipsHeight="55px"
        chipsWidth="250px"
        chipsClickable
        chipsLinkable
      >
        Shared With Me
      </MyItems>
    </div>
  );
};

export default page;
