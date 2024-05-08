"use client";

import React, {useState, useEffect} from "react";
import PatientPageHeader from "./components/PatientPageHeader";
import PatientInfo from "./components/PatientInfo";
import PatientEEG from "./components/PatientEEG";
import { Patient } from "@/app/interface";

interface PatientPageParms {
	params: {
		patient: string;
	};
}

const page = ({ params: { patient } }: PatientPageParms) => {
	const [patientInfo, setPatientInfo] = useState<Patient | null>(null);

	const getInfo = async () => {
		const res = await fetch(`http://localhost:3000/api/patient/get_patient/?id=${patient}`, {
      method: "GET",
      credentials: "include",
    });

		return await res.json();
	};

	const modifyAccess = async (newAcess: Array<string>) => {
		if (patientInfo) {
			setPatientInfo({...patientInfo, ...{access: newAcess}});
		}
	};

	useEffect(() => {
		getInfo().then((res) => setPatientInfo(res));
	}, []);

	return (
		<div className="flex flex-col my-3 mx-5 gap-8">
			<PatientPageHeader patient={patientInfo} modifyAccess={modifyAccess} />

			<div className="flex flex-col gap-8 mx-5">
				<PatientInfo patient={patientInfo} />
				<PatientEEG patient={patientInfo} />
			</div>
		</div>
	);
};

export default page;
