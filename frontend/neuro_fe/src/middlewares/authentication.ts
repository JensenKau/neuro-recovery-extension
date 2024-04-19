import { NextFetchEvent, NextRequest, NextResponse } from "next/server";
import { MiddlewareFactory } from "./types";
import { getApi } from "./api";

export const authenticationMiddleware: MiddlewareFactory = (next) => {
	return async (request: NextRequest, _next: NextFetchEvent) => {
		if (request.nextUrl.pathname.startsWith("/api")) {
			const access = request.cookies.get("jwt_access")?.value;
			const refresh = request.cookies.get("jwt_refresh")?.value;

			if (refresh === undefined) {
				return NextResponse.redirect(new URL("/login", request.url));
			}

			if (access === undefined) {
				const res = await fetch(`${getApi()}api/token/refresh/`, {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
					},
					body: JSON.stringify({
						refresh: refresh,
					}),
				});

				request.cookies.set("jwt_access", (await res.json()).access);
			}
		}

		return next(request, _next);
	};
};
