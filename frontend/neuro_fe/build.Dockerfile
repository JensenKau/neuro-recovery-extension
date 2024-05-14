FROM node:21.7.3-alpine as base
WORKDIR /app
ENV NEXT_PRIVATE_STANDALONE true
COPY package*.json ./
RUN npm ci
COPY . . 
RUN npm run build


FROM node:21.7.3-alpine as runtime
WORKDIR /app

ENV NODE_ENV=production

COPY --from=base /app/.next/standalone ./
COPY --from=base /app/.next/static ./.next/static

EXPOSE 3000
ENV PORT 3000

ENV HOSTNAME="0.0.0.0"