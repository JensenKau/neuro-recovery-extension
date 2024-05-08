export interface ShortPatient {
	id: number;
	name: string;
}

export interface Patient {
	id: number;
	owner: string;
	access: Array<string>;
	name: string;
	first_name: string;
	last_name: string;
	age: number;
	sex: "male" | "female";
	rosc: number;
	ohca: boolean;
	shockable_rhythm: boolean;
	ttm: number;
}


export interface ShortEEG {
	patient: number;
	name: string;
	created_at: string;
}