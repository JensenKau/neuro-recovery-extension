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
  path?: string;
  deletable: boolean;
  deleteHandler?: () => void;
  contentCentre? : boolean
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
  path,
  deletable,
  deleteHandler,
  contentCentre
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
            icon={Icon && !deletable? <Icon /> : undefined}
            label={
              additionalItem === undefined ? (
                item
              ) : (
                <div className="space-x-[370px]">
                  <span>{Icon ? <Icon/> : <span></span>} {additionalItem[index]}</span>
                  <span>{item}</span>

                </div>
              )
            }
            onClick={clicked}
            onDelete={deletable ? deleteHandler: undefined}
            deleteIcon={<DeleteIcon style={{color: "red"}}/>}
           
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
              justifyContent: contentCentre==undefined ? "space-between" : "center",
            }}
            clickable={canClick}
            component="a"
            href={linkable ? `${path}/${item}` : undefined}
          />
        ))}
      </div>
    </>
  );
};

export default Chips;
