"use client";
import React from "react";
import Chip from "@mui/material/Chip";
import Link from "next/link";
interface Props {
  items: string[];
  icon: React.ElementType;
  clicked?: () => void;
}

const Chips = ({ items, icon: Icon, clicked }: Props) => {
  return (
    <>
      {items.length === 0 && (
        <p className="flex justify-center items-center w-full">No item found</p>
      )}

      <div className="flex flex-wrap gap-2.5 ml-10 mr-5">
        {items.map((item, index) => (
          <Link key={item} href={`/home/${item}`}>
            <Chip
              key={index}
              icon={Icon ? <Icon /> : undefined}
              label={item}
              onClick={clicked}
              sx={{
                height: "55px",
                width: "250px",
                fontSize: "1.2rem",
                padding: "0 50px",
                backgroundColor: "#bae6fd",
                "&:hover": {
                  backgroundColor: "#38bdf8",
                },
                margin: "5px",
              }}
              clickable
            />
          </Link>
        ))}
      </div>
    </>
  );
};

export default Chips;
