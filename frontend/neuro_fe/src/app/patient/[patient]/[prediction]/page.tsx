"use client";

import React, { useEffect, useState } from "react";
import Tab from "../../../components/Tabs";
import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import SnackBarButton from "../../../components/SnackBarButton";
import { Margin } from "@mui/icons-material";
import Report from "./components/report/Report";
import { EEG } from "@/app/interface";

const page = () => {
  return (
    <div className="grid grid-cols-2 h-screen">
      <div />
      <Report />
    </div>
  );

  // return (
  //   <div>
  //     <div style={{ display: "flex", minHeight: "100vh" }}>
  //       <div style={{ flex: 1, padding: "20px", backgroundColor: "#f8f9fa" }}>
  //         <Tab />
  //       </div>
  //       <div style={{ flex: 1, padding: "20px", backgroundColor: "#d1d5db" }}>
  //         <h1 className="text-5xl mb-[10px] ml-[30px]">Prediction Report</h1>
  //         <div className="h-px bg-gray-400 w-full mb-[50px]"></div>
  //         <div className="flex justify-between m-[30px]">
  //           <div>
  //             <h2 className="text-3xl mb-[10px] text-blue-600">
  //               EEG information
  //             </h2>
  //             <p>start time: </p>
  //             <p>end time: </p>
  //             <p>utility frequency: </p>
  //             <p>sampling frequency: </p>
  //           </div>

  //           <div>
  //             <h2 className="text-3xl mb-[10px] text-blue-600">
  //               Model Prediction
  //             </h2>
  //             <p>model name: </p>
  //             <p>outcome prediction: </p>
  //             <p>cpc prediction: </p>
  //             <p>confidence: </p>
  //           </div>
  //         </div>

  //         <div className="m-[30px]">
  //           <div className="flex justify-between">
  //             <h2 className="text-3xl mb-[10px] text-blue-600">Report</h2>
  //           </div>
  //           <Box
  //             sx={{
  //               width: 625,
  //               maxWidth: "100%",
  //             }}
  //           >
  //             <TextField
  //               fullWidth
  //               label="Write Feedback Report Here"
  //               id="feedback report"
  //               multiline
  //               rows={4}
  //             />
  //           </Box>
  //           <div className="flex justify-end mt-[20px]">
  //             <SnackBarButton buttonInfo="Export As PDF">exported</SnackBarButton>
  //             <SnackBarButton buttonInfo="Save Changes">saved</SnackBarButton>
  //           </div>
  //         </div>
  //       </div>
  //     </div>
  //   </div>
  // );
};

export default page;
