"use client";
import React from "react";
import Chip from "@mui/material/Chip";

interface Props {
  items: string[];
  icon: React.ElementType;
  clicked?: () => void;
}

const Chips = ({ items, icon: Icon, clicked }: Props) => {
  return (
    <>
      {items.length === 0 && (
        <p
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            width: "100%",
          }}
        >
          No item found
        </p>
      )}

      <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", marginLeft: '40px', marginRight: '20px'}}>
        {items.map((item, index) => (
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
        ))}
      </div>
    </>
  );
};

export default Chips;
