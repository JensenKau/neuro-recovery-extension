export interface ShortPatient {
	id: number;
	name: string;
}

export interface Patient {
	id: number;
	owner: string;
	access: Array<string>;
	name: string;
	age: number;
	sex: "male" | "female";
	rosc: number;
	ohca: boolean;
	sr: boolean;
	ttm: number;
}