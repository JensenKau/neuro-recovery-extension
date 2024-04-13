"use client";
import React, { ComponentType } from "react";
import { useState } from "react";
import Chips from "./Chips";
import { FormProps } from "./PatientForm";

interface Props {
  children: string;
  childrenSize: string;
  initialItems: string[];
  FormButtonComponent?: ComponentType<FormProps>;
  FormButtonProps?: FormProps;
  chipsIcon?: React.ElementType;
  chipsHeight: string;
  chipsWidth: string;
  chipsClickable: boolean;
  chipsLinkable: boolean;
}

const MyItems = ({
  children,
  childrenSize,
  initialItems,
  FormButtonComponent,
  FormButtonProps,
  chipsIcon,
  chipsHeight,
  chipsWidth,
  chipsClickable,
  chipsLinkable,
}: Props) => {
  const [items, setItems] = useState<string[]>(initialItems);
  const handleFormSubmit = (name: string) => {
    setItems((currentItems) => [...currentItems, name]);
  };

  return (
    <div className="mt-[20px]">
      <div className="mb-[20px] flex justify-between">
        <span className={`text-blue-600 text-${childrenSize}`}>{children}</span>
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
      />
    </div>
  );
};

export default MyItems;
