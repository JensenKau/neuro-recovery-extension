"use client";
import React, { ComponentType } from "react";
import { useState } from "react";
import Chips from "./Chips";
import { FormProps } from "./PatientForm";

interface Props {
  children: string;
  initialItems: string[];
  FormButtonComponent?: ComponentType<FormProps>;
  FormButtonProps?: FormProps;
  chipsIcon?: React.ElementType;
  chipsHeight: string;
  chipsWidth: string;
  chipsClickable: boolean;
  chipsLinkable: boolean;
  chipsAdditionalInfo?: string[];
  chipsDeletable: boolean;
  chipsDeleteHandler?: () => void,
  chipsContentCenter?: boolean
  chipsExtraPath?: string
}

const MyItems = ({
  children,
  initialItems,
  FormButtonComponent,
  FormButtonProps,
  chipsIcon,
  chipsHeight,
  chipsWidth,
  chipsClickable,
  chipsLinkable,
  chipsAdditionalInfo,
  chipsDeletable,
  chipsDeleteHandler,
  chipsContentCenter,
  chipsExtraPath
}: Props) => {
  const [items, setItems] = useState<string[]>(initialItems);
  const handleFormSubmit = (name: string) => {
    setItems((currentItems) => [...currentItems, name]);
  };

  return (
    <div className="mt-[20px] mb-[50px]">
      <div className="mb-[20px] flex justify-between">
        <span className='text-blue-600' style={{fontSize: "35px"}}>{children}</span>
        {FormButtonComponent === undefined || FormButtonProps === undefined ? (
          <span></span>
        ) : (
          <FormButtonComponent
            {...FormButtonProps}
            onSubmit={handleFormSubmit}
          />
        )}
      </div>

      <Chips
        items={items}
        icon={chipsIcon}
        height={chipsHeight}
        width={chipsWidth}
        canClick={chipsClickable}
        linkable={chipsLinkable}
        additionalItem={chipsAdditionalInfo}
        deletable={chipsDeletable}
        deleteHandler={chipsDeleteHandler}
        contentCentre={chipsContentCenter}
        path={chipsExtraPath}
      />
    </div>
  );
};

export default MyItems;
