// Fastify server to serve static files from the static directory
import Fastify from 'fastify';
import path from 'path';
import fastifyStatic from '@fastify/static';
import { fileURLToPath } from 'url';
import { grpcClient, SearchPreset, type SearchRequest, type RecommendationRequest } from './grpc-client.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const fastify = Fastify({ logger: true });

// Manual CORS headers for API routes
fastify.addHook('preHandler', async (request, reply) => {
  if (request.url.startsWith('/api/')) {
    reply.header('Access-Control-Allow-Origin', '*');
    reply.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    reply.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    
    if (request.method === 'OPTIONS') {
      reply.send();
    }
  }
});

// Register static file serving
fastify.register(fastifyStatic, {
  root: path.join(__dirname, '../static'),
  prefix: '/', // Serve all files at root
  decorateReply: false
});

// API Routes

// Search movies endpoint
fastify.post<{
  Body: {
    query: string;
    options?: Partial<SearchRequest>;
  };
}>('/api/search', async (request, reply) => {
  try {
    const { query, options = {} } = request.body;
    
    if (!query || query.trim() === '') {
      return reply.code(400).send({ error: 'Query is required' });
    }

    const results = await grpcClient.searchMovies(query, options);
    return { results, success: true };
  } catch (error) {
    fastify.log.error(error);
    return reply.code(500).send({ 
      error: error instanceof Error ? error.message : 'Search failed',
      success: false 
    });
  }
});

// Get movie recommendations endpoint
fastify.post<{
  Body: {
    imdbId: string;
    options?: Partial<RecommendationRequest>;
  };
}>('/api/recommendations', async (request, reply) => {
  try {
    const { imdbId, options = {} } = request.body;
    
    if (!imdbId || imdbId.trim() === '') {
      return reply.code(400).send({ error: 'IMDb ID is required' });
    }

    const results = await grpcClient.getRecommendations(imdbId, options);
    return { results, success: true };
  } catch (error) {
    fastify.log.error(error);
    return reply.code(500).send({ 
      error: error instanceof Error ? error.message : 'Recommendations failed',
      success: false 
    });
  }
});

// Get search presets endpoint
fastify.get('/api/presets', async (request, reply) => {
  return {
    presets: {
      PRESET_UNSPECIFIED: SearchPreset.PRESET_UNSPECIFIED,
      PRESET_POPULAR: SearchPreset.PRESET_POPULAR,
      PRESET_NICHE: SearchPreset.PRESET_NICHE,
      PRESET_CULTURAL: SearchPreset.PRESET_CULTURAL
    },
    success: true
  };
});

// Health check endpoint
fastify.get('/api/health', async (request, reply) => {
  return { status: 'ok', timestamp: new Date().toISOString() };
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
