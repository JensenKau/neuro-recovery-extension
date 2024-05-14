import { NextFetchEvent, NextRequest, NextResponse } from "next/server";
import { MiddlewareFactory } from "./types";
import { getApi } from "./api";

export const headerMiddleware: MiddlewareFactory = (next) => {
	const extractBody = async (request: NextRequest) => {
		if (request.method === "POST") {
			const clone = request.clone();

			return await request.blob();
		}
		return undefined;
	};

	return async (request: NextRequest, _next: NextFetchEvent) => {
		if (request.nextUrl.pathname.startsWith("/api")) {
			const accessToken = request.cookies.get("jwt_access")?.value;
			request.headers.append("Authorization", `Bearer ${accessToken}`);

			const res = await fetch(`${getApi()}${request.nextUrl.pathname}/${request.nextUrl.searchParams.toString() !== "" ? "?" + request.nextUrl.searchParams.toString() : ""}`, {
				method: request.method,
				headers: request.headers,
				body: await extractBody(request),
			});

			const content = await res.blob();
			const output = new NextResponse(content, { status: res.status });

			if (accessToken !== undefined) {
				output.cookies.set({ name: "jwt_access", value: accessToken });
			}

			return output;
		}

		return next(request, _next);
	};
};
