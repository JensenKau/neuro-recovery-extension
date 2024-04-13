"use client";
import React from "react";
import Chip from "@mui/material/Chip";
import Link from "next/link";
import { ClickAwayListener } from "@mui/material";
interface Props {
  items: string[];
  icon?: React.ElementType;
  clicked?: () => void;
  height: string;
  width: string;
  canClick: boolean;
  linkable: boolean;
}

const Chips = ({
  items,
  icon: Icon,
  clicked,
  height,
  width,
  canClick,
  linkable,
}: Props) => {
  return (
    <>
      {items.length === 0 && (
        <p className="flex justify-center items-center w-full">No item found</p>
      )}

      <div className="flex flex-wrap gap-2.5">
        {items.map((item, index) => (
          <Chip
            key={index}
            icon={Icon ? <Icon /> : undefined}
            label={item}
            onClick={clicked}
            sx={{
              height: { height },
              width: { width },
              fontSize: "1.2rem",
              padding: canClick ? "0 20px" : "0",
              backgroundColor: "#bae6fd",
              "&:hover": linkable
                ? {
                    backgroundColor: "#64b5f6",
                  }
                : {},
              margin: canClick ? "5px" : "0",
              justifyContent: canClick ? "center" : "flex-start",
            }}
            clickable={canClick}
            component="a"
            href={linkable ? `/home/${item}` : undefined}
          />
        ))}
      </div>
    </>
  );
};

export default Chips;
