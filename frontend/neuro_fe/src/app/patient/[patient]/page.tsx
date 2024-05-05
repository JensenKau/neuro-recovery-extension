"use client";

import React from "react";
import HomeIcon from "./components/HomeIcon";
import MyItems from "../../components/MyItems";
import EditIcon from "@mui/icons-material/Edit";
import CustomButton from "../../components/CustomButton";
import PatientForm from "../../components/PatientForm";
import ShareIcon from "@mui/icons-material/Share";
import ShareForm from "../../components/ShareProfile";
import InfoIcon from "@mui/icons-material/Feed";
import PatientPageHeader from "./components/PatientPageHeader";
import PatientInfo from "./components/PatientInfo";
import PatientEEG from "./components/PatientEEG";

interface PatientPageParms {
	params: {
		patient: string;
	};
}

const page = ({ params: { patient } }: PatientPageParms) => {
	const patientId = parseInt(patient);
	const patientName = parseInt(patient);

	return (
		<div className="flex flex-col my-3 mx-5 gap-8">
			<PatientPageHeader />

			<div className="flex flex-col gap-8 mx-5">
				<PatientInfo />
				<PatientEEG />
			</div>
		</div>
	);

	// return (
	// <div className="mt-[50px] mb-[40px] ml-[50px] mr-[50px]">
	//   <div className="flex justify-between mb-[30px]">
	//     <div className="flex">
	//       <HomeIcon />
	//       <div className="mt-6 ml-5 text-5xl">
	//         My Patients {">"}{" "}
	//         <span className="text-blue-600">{patientName}</span>
	//       </div>
	//     </div>

	//       <div className="mt-[30px]">
	//         <ShareForm
	//           title="Share Profile"
	//           ButtonComponent={CustomButton}
	//           buttonProps={{
	//             children: "share profile",
	//             icon: ShareIcon,
	//             style: "text",
	//           }}
	//           submitFormInfo="done"
	//         />
	//       </div>
	//     </div>
	//     <div className="h-px bg-blue-400 w-full mb-[50px]"></div>
	//     <MyItems
	//       initialItems={[
	//         "First Name: ",
	//         "Last Name: ",
	//         "Age: ",
	//         "Gender: ",
	//         "ROSC: ",
	//         "OHCA: ",
	//         "Shockable Rhythm: ",
	//         "TTM: ",
	//       ]}
	//       FormButtonComponent={PatientForm}
	//       FormButtonProps={{
	//         title: "Edit Patient Infomation",
	//         ButtonComponent: CustomButton,
	//         buttonProps: {
	//           children: "Edit patient info",
	//           icon: EditIcon,
	//           style: "outlined",
	//         },
	//         submitFormInfo: "save changes",
	//       }}
	//       chipsHeight="55px"
	//       chipsWidth="327px"
	//       chipsClickable={false}
	//       chipsLinkable={false}
	//       chipsDeletable={false}
	//     >
	//       Pateint Information
	//     </MyItems>

	//     <MyItems
	//       initialItems={["Dynamic Model", "Static Model"]}
	//       FormButtonComponent={PatientForm}
	//       FormButtonProps={{
	//         title: "Edit Patient Infomation",
	//         ButtonComponent: CustomButton,
	//         buttonProps: {
	//           children: "Generate EEG",
	//           style: "outlined",
	//         },
	//         submitFormInfo: "save changes",
	//       }}
	//       chipsIcon={InfoIcon}
	//       chipsHeight="55px"
	//       chipsWidth="1350px"
	//       chipsClickable
	//       chipsLinkable={true}
	//       chipsAdditionalInfo={["[12/03/2024 00:00:00]", "[20/04/2024 00:00:00]"]}
	//       chipsDeletable = {true}
	//       chipsDeleteHandler={() => console.log("hi")}
	//       chipsExtraPath={patientName}
	//     >
	//       Patients EEGs
	//     </MyItems>
	//   </div>
	// );
};

export default page;
