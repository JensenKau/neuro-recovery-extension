"use client";

import {
	Dialog,
	DialogTitle,
	Typography,
	TextField,
	DialogContent,
	Button,
} from "@mui/material";
import { Add } from "@mui/icons-material";
import React from "react";
import ShareProfileItem from "./ShareProfileItem";

interface ShareProfileForm {
	open: boolean;
	onClose(value: boolean): void;
}

const ShareProfileForm = ({ open, onClose }: ShareProfileForm) => {
	return (
		<Dialog
			open={open}
			onClose={() => onClose(false)}
			fullWidth={true}
			maxWidth="sm"
		>
			<DialogTitle className="mx-auto mt-5 mb-3">
				<Typography variant="h5" className="text-[#01579b]">
					New Patient
				</Typography>
			</DialogTitle>

			<DialogContent className="flex flex-col gap-4 w-full">
				<div className="flex gap-3 my-2">
					<TextField className="w-full" label="Share with..." />
					<Button variant="contained" className="h-auto w-fit flex gap-1">
            <Add className="text-bold" />
            Add
          </Button>
				</div>

				<div className="flex flex-col gap-1">
					<Typography className="text-blue-600 text-xl">Owner</Typography>
					<ShareProfileItem email="jkau0039@student.monash.edu" />
				</div>

				<div className="flex flex-col gap-1">
					<Typography className="text-blue-600 text-xl">
						User with Access
					</Typography>
					<div className="flex flex-col gap-2 overflow-y-scroll overflow-x-hidden h-[200px]">
						<ShareProfileItem email="jkau0039@student.monash.edu" deletable />
						<ShareProfileItem email="jkau0039@student.monash.edu" deletable />
            <ShareProfileItem email="jkau0039@student.monash.edu" deletable />
            <ShareProfileItem email="jkau0039@student.monash.edu" deletable />
            <ShareProfileItem email="jkau0039@student.monash.edu" deletable />
            <ShareProfileItem email="jkau0039@student.monash.edu" deletable />
            <ShareProfileItem email="jkau0039@student.monash.edu" deletable />
					</div>
				</div>

        <Button variant="contained">Done</Button>
			</DialogContent>
		</Dialog>
	);
};

export default ShareProfileForm;
