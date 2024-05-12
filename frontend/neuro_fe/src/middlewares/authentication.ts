import { NextFetchEvent, NextRequest, NextResponse } from "next/server";
import { MiddlewareFactory } from "./types";
import { getApi } from "./api";
import { jwtDecode } from "jwt-decode";

const freeRoute = [
	"/",
	"/login",
	"/register"
]

export const authenticationMiddleware: MiddlewareFactory = (next) => {
	return async (request: NextRequest, _next: NextFetchEvent) => {
		if (!freeRoute.includes(request.nextUrl.pathname)) {
			const access = request.cookies.get("jwt_access")?.value;
			const refresh = request.cookies.get("jwt_refresh")?.value;

			const tokenRefresh = async () => {
				const res = await fetch(`${getApi()}/api/token/refresh/`, {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
					},
					body: JSON.stringify({
						refresh: refresh,
					}),
				});
				request.cookies.set("jwt_access", (await res.json()).access);
			};

			if (refresh !== undefined) {
				const refreshPayload = jwtDecode(refresh);
				if (refreshPayload.exp === undefined || refreshPayload.exp * 1000 < Date.now()) {
					return NextResponse.redirect(new URL("/login", request.url));
				}
			} else {
				return NextResponse.redirect(new URL("/login", request.url));
			}

			if (access !== undefined) {
				const accessPayload = jwtDecode(access);
				if (accessPayload.exp === undefined || accessPayload.exp * 1000 < Date.now()) {
					await tokenRefresh();
				}
			} else {
				await tokenRefresh();
			}
		}

		return next(request, _next);
	};
};
