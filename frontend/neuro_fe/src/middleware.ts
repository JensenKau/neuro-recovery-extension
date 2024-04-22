import { stackMiddlewares } from "./middlewares/stack";
import { authenticationMiddleware } from "./middlewares/authentication";
import { loginMiddleware } from "./middlewares/login";
import { signupMiddleware } from "./middlewares/signup";
import { headerMiddleware } from "./middlewares/header";

export default stackMiddlewares(
  [
    loginMiddleware, 
    signupMiddleware, 
    authenticationMiddleware,
    headerMiddleware,
  ]
);