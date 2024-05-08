import React from "react";
import IconButton from "@mui/material/IconButton";
import HomeButton from "@mui/icons-material/Home";
import Link from "@mui/material/Link";

const HomeIcon = () => {
	return (
		<IconButton className="hover:bg-[#d1d5db]">
			<Link href="/home">
				<HomeButton className="text-7xl" />
			</Link>
		</IconButton>
	);
};

export default HomeIcon;
