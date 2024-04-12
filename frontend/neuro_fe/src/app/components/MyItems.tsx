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
  chipsHeight: string
  chipsWidth: string
  chipsClickable: boolean
}

const MyItems = ({
  children,
  initialItems,
  FormButtonComponent,
  FormButtonProps,
  chipsIcon,
  chipsHeight,
  chipsWidth,
  chipsClickable
}: Props) => {
  const [items, setItems] = useState<string[]>(initialItems);
  const handleFormSubmit = (name: string) => {
    setItems((currentItems) => [...currentItems, name]);
  };

  return (
    <div className="mt-[50px]">
      <div className="ml-[45px] mb-[20px] flex justify-between">
        <span className="text-blue-600 text-3xl">{children}</span>
        {FormButtonComponent === undefined || FormButtonProps === undefined ? (
          <span></span>
        ) : (
          <FormButtonComponent
            {...FormButtonProps}
            onSubmit={handleFormSubmit}
          />
        )}
      </div>

      <Chips items={items} icon={chipsIcon} height={chipsHeight} width={chipsWidth} clickable={chipsClickable}/>
    </div>
  );
};

export default MyItems;
