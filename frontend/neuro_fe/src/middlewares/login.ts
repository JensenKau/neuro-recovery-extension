import { NextFetchEvent, NextRequest, NextResponse } from "next/server";
import { MiddlewareFactory } from "./types";
import { getApi } from "./api";
import { jwtDecode } from "jwt-decode";

export const loginMiddleware: MiddlewareFactory = (next) => {
	return async (request: NextRequest, _next: NextFetchEvent) => {
		if (request.nextUrl.pathname === "/api/login") {
			const requestBody = await request.json();

			const res = await fetch(`${getApi()}/api/token/`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify(requestBody)
			});

			const content = await res.json();
			const access_payload = jwtDecode(content.access);
			const refresh_payload = jwtDecode(content.refresh);

			const output = NextResponse.next();

			const access_cookie = { name: "jwt_access", value: content.access };
			const refresh_cookie = { name: "jwt_refresh", value: content.refresh };				

			if (requestBody.rememberme && access_payload.exp !== undefined && refresh_payload.exp !== undefined) {
				output.cookies.set({ ...access_cookie, expires: access_payload.exp * 1000 });
				output.cookies.set({ ...refresh_cookie, expires: refresh_payload.exp * 1000 });
			} else {
				output.cookies.set(access_cookie);
				output.cookies.set(refresh_cookie);
			}

			return output;
		}

		return next(request, _next);
	};
};
