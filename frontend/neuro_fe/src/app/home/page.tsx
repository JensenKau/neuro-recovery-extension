"use client";

import React from "react";
import { useState, useEffect } from "react";
import { ShortPatient, User } from "../interface";
import { unstable_noStore as noStore } from "next/cache";

import HomeSection from "./components/HomeSection";
import HomeSectionHeader from "./components/HomeSectionHeader";
import AddPatientButton from "./components/AddPatientButton";

const HomePage = () => {
  noStore();

  const [owned, setOwned] = useState<ShortPatient[]>([]);
  const [access, setAccess] = useState<ShortPatient[]>([]);
  const [user, setUser] = useState<User>();

  const getPatients = async (): Promise<{
    owned: ShortPatient[];
    access: ShortPatient[];
  }> => {
    const res = await fetch("http://localhost:3000/api/patient/get_patients/", {
      method: "GET",
      credentials: "include",
    });
    return await res.json();
  };

  const getUser = async (): Promise<User> => {
    const res = await fetch("http://localhost:3000/api/user/get_user/", {
      method: "GET",
      credentials: "include"
    });
    return await res.json();
  }

  useEffect(() => {
    getPatients().then((value) => {
      setOwned(value.owned);
      setAccess(value.access);
    });

    getUser().then(setUser);
  }, []);

  return (
    <div className="mt-[60px] mb-[50px] ml-[50px] mr-[50px]">
      <div className="mb-[30px] text-5xl">
        Welcome Back, {" "}
        <span className="text-blue-600">
          {user && user.role === "doctor" ? "Dr. " : ""}
          {user && user.fullname}
        </span>
      </div>

      <div className="grid grid-cols-2">
        <HomeSectionHeader label="My Patients" />
        <AddPatientButton className="w-1/4 ml-auto" onSubmit={(value) => setOwned([...owned, value])} />
      </div>

      <HomeSection
        patients={owned}
        message="You do not seem to have any patients"
      />

      <HomeSectionHeader label="Shared with Me" className="mt-16 mb-5" />
      <HomeSection
        patients={access}
        message="You do not seem to have any patients shared with you"
      />
    </div>
  );
};

export default HomePage;
