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


export interface EEG {
	patient: number;
	name: string;
	start_time: number;
	end_time: number;
	utility_freq: number;
	sampling_freq: number;
	created_at: string;
	updated_at: string;
}