"use client";
import * as React from "react";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import Image from "next/image";
import eegData from "../img/eeg.jpeg";
import BCMData from "../img/brain_connectivity_mapping.jpeg";
import FCData from "../img/functional_connectivity.jpeg";

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function CustomTabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          <Typography>{children}</Typography>
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    "aria-controls": `simple-tabpanel-${index}`,
  };
}

export default function BasicTabs() {
  const [value, setValue] = React.useState(0);

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  return (
    <Box sx={{ width: "100%" }}>
      <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Tabs
          value={value}
          onChange={handleChange}
          aria-label="basic tabs example"
          textColor="primary"
          indicatorColor="primary"
          centered
          variant="fullWidth"
        >
          <Tab label="EEG" {...a11yProps(0)} />
          <Tab label="Brain Connectivity Mapping" {...a11yProps(1)} />
          <Tab label="Functional Connectivity" {...a11yProps(2)} />
        </Tabs>
      </Box>
      <CustomTabPanel value={value} index={0}>
        <Image
          src={eegData}
          alt="EEG DATA"
          width={500}
          height={300}
          layout="responsive"
        />
      </CustomTabPanel>
      <CustomTabPanel value={value} index={1}>
        <Image
          src={BCMData}
          alt="BRAIN CONNECTIVITY MAPPING OF THE PATIENT"
          width={900}
          height={300}
        />
      </CustomTabPanel>
      <CustomTabPanel value={value} index={2}>
        <Image
          src={FCData}
          alt="FUCTIONAL CONNECTIVITY OF THE PATIENT"
          width={900}
          height={300}
        />
      </CustomTabPanel>
    </Box>
  );
}
