import { stackMiddlewares } from "./middlewares/stack";
import { authenticationMiddleware } from "./middlewares/authentication";
import { loginMiddleware } from "./middlewares/login";
import { signupMiddleware } from "./middlewares/signup";
import { headerMiddleware } from "./middlewares/header";
import { routeMiddleware } from "./middlewares/route";

export default stackMiddlewares(
  [
    loginMiddleware, 
    signupMiddleware, 
    authenticationMiddleware,
    headerMiddleware,
    routeMiddleware
  ]
);

export const config = {
  matcher: [
    '/((?!_next|api/auth).*)(.+)'
  ],
}