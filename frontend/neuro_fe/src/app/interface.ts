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
	id: number;
	patient: number;
	name: string;
	start_time: number;
	end_time: number;
	utility_freq: number;
	sampling_freq: number;
	created_at: string;
	updated_at: string;
}


export interface Prediction {
	id: number;
	patient_eeg: number;
	ai_model: {
		id: number;
		name: string;
	};
	outcome_pred: string;
	confidence: number;
	comments: string;
}