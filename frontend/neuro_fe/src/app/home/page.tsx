"use client";

import React from "react";
import { useState, useEffect } from "react";
import MyItems from "../components/MyItems";
import AddIcon from "@mui/icons-material/Add";
import CustomButton from "../components/CustomButton";
import PatientForm from "../components/PatientForm";
import Folder from "@mui/icons-material/Folder";
import { ShortPatient } from "./interface";
import { unstable_noStore as noStore } from "next/cache";

import HomeSection from "./components/HomeSection";
import HomeSectionHeader from "./components/HomeSectionHeader";

const HomePage = () => {
	noStore();

	const [owned, setOwned] = useState<ShortPatient[]>([]);
	const [access, setAccess] = useState<ShortPatient[]>([]);

	const func = async (): Promise<{
		owned: ShortPatient[];
		access: ShortPatient[];
	}> => {
		const res = await fetch("http://localhost:3000/api/patient/get_patients/", {
			method: "GET",
			credentials: "include",
		});

		return await res.json();
	};

	useEffect(() => {
		func().then((value) => {
			setOwned(value.owned), setAccess(value.access);
		});
	}, []);

	return (
		<div className="mt-[60px] mb-[50px] ml-[50px] mr-[50px]">
			<div className="mb-[80px] text-5xl">
				Welcome Back, <span className="text-blue-600">Doctor</span>
			</div>

			<HomeSectionHeader label="My Patients" />
			<HomeSection patients={owned} message="You do not seem to have any patients" />

			<HomeSectionHeader label="Shared with Me" className="mt-16 mb-5" />
			<HomeSection patients={access} message="You do not seem to have any patients shared with you" />
		</div>
	);
};

export default HomePage;
