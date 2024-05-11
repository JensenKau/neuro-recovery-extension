import { NextFetchEvent, NextRequest, NextResponse } from "next/server";
import { MiddlewareFactory } from "./types";
import { getApi } from "./api";
import { jwtDecode } from "jwt-decode";

const strictRoute = [
  { path: new RegExp("/patient/.*"), validator: "" }
]

export const routeMiddleware: MiddlewareFactory = (next) => {
  return async (request: NextRequest, _next: NextFetchEvent) => {
    const routeCheck = strictRoute.reduce((total, currVal) => {
      if (!total) {
        return Boolean(request.nextUrl.pathname.match(currVal.path));
      }
      return true;
    }, false);

    if (routeCheck && !request.nextUrl.pathname.startsWith("/api")) {
      const accessToken = request.cookies.get("jwt_access")?.value;

      const value = request.nextUrl.pathname.split("/");

      const res = await fetch(`${getApi()}/api/user/patient_access/?patient_id=${value[2]}`, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${accessToken}`,
        }
      });

      if (res.status !== 200) {
        return NextResponse.redirect(new URL("/home", request.url));
      }
    }

    return next(request, _next);
  };
};