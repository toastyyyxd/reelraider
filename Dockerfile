# Use official Node.js LTS image
FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source and build
COPY . .

# Build TypeScript (if not already built)
RUN npm run build

EXPOSE 8000

# Start the server (using the built JS output)
CMD ["node", "dist/server/index.js"]