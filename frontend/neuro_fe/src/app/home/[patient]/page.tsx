import React from "react";
import HomeIcon from "../../components/HomeIcon";
import MyItems from "../../components/MyItems";
import EditIcon from "@mui/icons-material/Edit";
import CustomButton from "../../components/CustomButton";
import PatientForm from "../../components/PatientForm";

const page = ({ params }: any) => {
  const patientName = decodeURIComponent(params.patient);
  return (
    <div>
      <div className="mt-[30px] ml-[30px] flex">
        <HomeIcon />
        <div className="mt-6 ml-5 text-5xl">
          My Patients {">"} <span className="text-blue-600">{patientName}</span>
        </div>
      </div>
      <MyItems
        initialItems={[
          "First Name: ",
          "Last Name: ",
          "Age: ",
          "Gender: ",
          "ROSC: ",
          "OHCA: ",
          "Shockable Rhythm: ",
          "TTM: ",
        ]}
        FormButtonComponent={PatientForm}
        FormButtonProps={{
          title: "Edit Patient Infomation",
          ButtonComponent: CustomButton,
          buttonProps: { children: "Edit", icon: EditIcon },
          submitFormInfo: "save changes",
        }}
        chipsHeight="55px"
        chipsWidth="320px"
        chipsClickable={false}
      >
        Pateint Information
      </MyItems>
    </div>
  );
};

export default page;
