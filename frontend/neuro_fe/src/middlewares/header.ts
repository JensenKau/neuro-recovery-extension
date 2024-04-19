import { NextFetchEvent, NextRequest, NextResponse } from "next/server";
import { MiddlewareFactory } from "./types";
import { getApi } from "./api";

export const headerMiddleware: MiddlewareFactory = (next) => {
  return async (request: NextRequest, _next: NextFetchEvent) => {
		if (request.nextUrl.pathname.startsWith("/api")) {
			const accessToken = request.cookies.get("jwt_access")?.value;

			const res = await fetch(`${getApi()}${request.nextUrl.pathname}/`, {
				method: request.method,
				headers: {
					"Content-Type": "application/json",
					"Authorization": `Bearer ${accessToken}`
				},
				body: (request.method === "POST") ? await request.text() : undefined
			});

			const content = await res.json();
			const output = NextResponse.json(content);

			if (accessToken !== undefined) {
				output.cookies.set({ name: "jwt_access", value: accessToken});
			}

			return output;
		}

		return next(request, _next);
	}
}