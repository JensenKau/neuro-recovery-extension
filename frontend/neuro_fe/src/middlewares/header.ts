import { NextFetchEvent, NextRequest, NextResponse } from "next/server";
import { MiddlewareFactory } from "./types";
import { getApi } from "./api";

export const headerMiddleware: MiddlewareFactory = (next) => {
  return async (request: NextRequest, _next: NextFetchEvent) => {
		if (request.nextUrl.pathname.startsWith("/api")) {
			const res = await fetch(`${getApi()}${request.nextUrl.pathname}/`, {
				method: request.method,
				headers: {
					"Content-Type": "application/json",
					"Authorization": `Bearer ${request.cookies.get("jwt_access")}`
				},
				body: await request.text()
			});

			const content = await res.json();

			return NextResponse.json(content);
		}

		return next(request, _next);
	}
}