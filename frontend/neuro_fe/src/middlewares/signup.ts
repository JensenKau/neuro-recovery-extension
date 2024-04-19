import { NextFetchEvent, NextRequest, NextResponse } from "next/server";
import { MiddlewareFactory } from "./types";
import { getApi } from "./api";

export const signupMiddleware: MiddlewareFactory = (next) => {
	return async (request: NextRequest, _next: NextFetchEvent) => {
		if (request.nextUrl.pathname === "api/signup/") {
			const res = await fetch(`${getApi()}api/user/create_user/`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: await request.text(),
			});

			return NextResponse.next();
		}

		return next(request, _next);
	};
};
