"use client";

import React from "react";

interface HomeSectionHeaderProps {
	label: string;
	className?: string;
}

const HomeSectionHeader = ({ label, className = "" }: HomeSectionHeaderProps) => {
	return (
		<div className={`text-blue-600 text-3xl ${className}`}>
			{label}
		</div>
	)
};

export default HomeSectionHeader;