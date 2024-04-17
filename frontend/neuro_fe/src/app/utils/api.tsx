"use client";

import { jwtDecode } from "jwt-decode";

const getApi = () => {
	return "http://localhost:8000/";
}

const validateToken = async () => {
	if (typeof window !== "undefined") {
		const rememberme = window.localStorage.getItem("rememberme");

		if (rememberme !== null) {
			const token = (rememberme === "true") ? window.localStorage.getItem("jwt_token") : window.sessionStorage.getItem("jwt_token");

			if (token !== null) {
				const payload = JSON.parse(token);

				const access_payload = jwtDecode(payload.access);
				const refresh_payload = jwtDecode(payload.refresh);
				const curr_time = (new Date()).getTime();

				if (access_payload.exp !== undefined && access_payload.exp * 1000 < curr_time) {
					if (refresh_payload.exp !== undefined && refresh_payload.exp * 1000 > curr_time) {
						const res = await fetch(
							`${getApi()}token/refresh/`,
							{
								method: "POST",
								headers: {
									"Content-Type": "application/json"
								},
								body:
									JSON.stringify({
										"refresh": payload.refresh
									})
							}
						);

						if (rememberme === "true") {
							window.localStorage.setItem("jwt_token", JSON.stringify((await res.json()).access));
						} else {
							window.sessionStorage.setItem("jwt_token", JSON.stringify((await res.json()).access));
						}

						return true;
					}
				}
			}
		}
	}

	return false;
};


export const apiCall = async (method: "GET" | "POST", urlpath: string, body?: any) => {
	const validated = await validateToken();

	if (validated && typeof window !== "undefined") {
		const rememberme = window.localStorage.getItem("rememberme");

		if (rememberme !== null) {
			const token = (rememberme === "true") ? window.localStorage.getItem("jwt_token") : window.sessionStorage.getItem("jwt_token");

			if (token !== null) {
				const accessToken = JSON.parse(token).access;

				return await fetch(
					`${getApi()}${urlpath}`,
					{
						method: method,
						headers: {
							"Content-Type": "application/json",
							"Authorization": `Bearer ${accessToken}`
						},
						body: body
					}
				)
			}
		}
	}

	return undefined;
};


export const apiSignIn = async (rememberme: boolean, email: string, password: string) => {
	if (typeof window !== "undefined") {
		const res = await fetch(
			`${getApi()}api/token/`,
			{
				method: "POST",
				headers: {
					"Content-Type": "application/json"
				},
				body:
					JSON.stringify({
						"email": email,
						"password": password
					})
			}
		);

		window.localStorage.setItem("rememberme", rememberme.toString());

		if (rememberme) {
			window.localStorage.setItem("jwt_token", await res.text());
		} else {
			window.sessionStorage.setItem("jwt-token", await res.text());
		}
	}
};