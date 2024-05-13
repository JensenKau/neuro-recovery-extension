FROM node:21.7.3-alpine as base
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . . 
RUN npm run build


FROM node:21.7.3-alpine as runtime
WORKDIR /app

ENV NODE_ENV=production

COPY --from=base /app/.next/standalone ./

EXPOSE 3000
ENV PORT 3000

CMD HOSTNAME="0.0.0.0" node server.js