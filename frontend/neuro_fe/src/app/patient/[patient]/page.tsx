"use client";

import React, {useState, useEffect} from "react";
import PatientPageHeader from "./components/header/PatientPageHeader";
import PatientInfo from "./components/info/PatientInfo";
import PatientEEG from "./components/eeg/PatientEEG";
import { Patient } from "@/app/interface";
import { ToastContainer } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';

interface PatientPageParms {
	params: {
		patient: string;
	};
}

const page = ({ params: { patient } }: PatientPageParms) => {
	const [patientInfo, setPatientInfo] = useState<Patient | null>(null);
	const [update, setUpdate] = useState(false);

	const getInfo = async () => {
		const res = await fetch(`http://localhost:3000/api/patient/get_patient/?id=${patient}`, {
      method: "GET",
      credentials: "include",
    });
		return await res.json();
	};

	useEffect(() => {
		getInfo().then((res) => setPatientInfo(res));
	}, []);

	useEffect(() => {
		getInfo().then((res) => setPatientInfo(res));
		setUpdate(false);
	}, [update])

	return (
		<div className="flex flex-col my-3 mx-5 gap-8">
			<PatientPageHeader patient={patientInfo} onUpdate={setUpdate} />

			<div className="flex flex-col gap-8 mx-5">
				<PatientInfo patient={patientInfo} />
				<PatientEEG patient={patientInfo} />
			</div>
		</div>
	);
};

export default page;
