# Use official Node.js LTS image
FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install --production || true

COPY src ./src

EXPOSE 8000

CMD ["node", "src/server/app-fastify.js"]
