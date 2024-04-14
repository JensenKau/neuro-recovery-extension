"use client";
import React from "react";
import Chip from "@mui/material/Chip";
import DeleteIcon from '@mui/icons-material/Delete';

interface Props {
  items: string[];
  additionalItem?: string[];
  icon?: React.ElementType;
  clicked?: () => void;
  height: string;
  width: string;
  canClick: boolean;
  linkable: boolean;
  deletable: boolean;
  deleteHandler?: () => void;
}

const Chips = ({
  items,
  additionalItem,
  icon: Icon,
  clicked,
  height,
  width,
  canClick,
  linkable,
  deletable,
  deleteHandler
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
            label={
              additionalItem === undefined ? (
                item
              ) : (
                <div className="space-x-[370px]">
                  <span>{additionalItem[index]}</span>
                  <span>{item}</span>
                </div>
              )
            }
            onClick={clicked}
            onDelete={deletable ? deleteHandler: undefined}
            deleteIcon={<DeleteIcon color="error"/>}
           
            sx={{
              height: { height },
              width: { width },
              fontSize: "1.2rem",
              padding: canClick ? "0 20px" : "0 20px",
              backgroundColor: "#bae6fd",
              "&:hover": canClick
                ? {
                    backgroundColor: "#64b5f6",
                  }
                : {},
              margin: canClick ? "5px" : "0",
              justifyContent: linkable ? "center" : "space-between",
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
