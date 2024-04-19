import { NextFetchEvent, NextRequest, NextResponse } from "next/server";
import { MiddlewareFactory } from "./types";
import { getApi } from "./api";
import { jwtDecode } from "jwt-decode";

export const loginMiddleware: MiddlewareFactory = (next) => {
	return async (request: NextRequest, _next: NextFetchEvent) => {
		if (request.nextUrl.pathname === "/api/login/") {
			const res = await fetch(`${getApi()}/api/token/`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: await request.text()
			});

			const content = await res.json();
			const access_payload = jwtDecode(content.access);
			const refresh_payload = jwtDecode(content.refresh);

			const output = NextResponse.next();

			if (access_payload.exp !== undefined && refresh_payload.exp !== undefined) {
				output.cookies.set({
					name: "jwt_access",
					value: content.access,
					expires: access_payload.exp * 1000,
				});
	
				output.cookies.set({
					name: "jwt_refresh",
					value: content.refresh,
					expires: refresh_payload.exp * 1000,
				});
			}

			return output;
		}

		return next(request, _next);
	};
};
