// Fastify server to serve static files from the static directory
import Fastify from 'fastify';
import path from 'path';
import fastifyStatic from '@fastify/static';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const fastify = Fastify({ logger: true });

fastify.register(fastifyStatic, {
  root: path.join(__dirname, '../static'),
  prefix: '/', // Serve all files at root
  decorateReply: false
});
const start = async () => {
  try {
    await fastify.listen({ port: 8000, host: '0.0.0.0' });
    fastify.log.info(`Server listening on http://localhost:8000`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();
