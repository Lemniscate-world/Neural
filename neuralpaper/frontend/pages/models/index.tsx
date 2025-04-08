import { useState, useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { motion } from 'framer-motion';
import axios from 'axios';

// Model card interface
interface ModelCard {
  id: string;
  name: string;
  description: string;
  image: string;
  category: string;
  complexity: string;
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelCard[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Fetch models
  useEffect(() => {
    const fetchModels = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // Fetch models from the API
        const response = await axios.get('/api/models');
        const modelsData = response.data.models;

        // Add category and complexity if not present
        const processedModels = modelsData.map((model: any) => ({
          ...model,
          category: model.category || 'Uncategorized',
          complexity: model.complexity || 'Medium',
          image: model.image || `/models/${model.id}.png`
        }));

        setModels(processedModels);
        setIsLoading(false);
      } catch (err: any) {
        setError(err.response?.data?.detail || err.message || 'Failed to load models');
        setIsLoading(false);
      }
    };

    fetchModels();
  }, []);

  // Filter and search models
  const filteredModels = models.filter(model => {
    // Apply category filter
    if (filter !== 'all' && model.category !== filter) {
      return false;
    }

    // Apply search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        model.name.toLowerCase().includes(query) ||
        model.description.toLowerCase().includes(query)
      );
    }

    return true;
  });

  // Get unique categories for filter
  const uniqueCategories = Array.from(new Set(models.map(model => model.category)));
  const categories = ['all', ...uniqueCategories];

  if (isLoading) {
    return (
      <div className="min-h-screen bg-neural-dark text-white flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-neural-secondary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-neural-dark text-white flex flex-col items-center justify-center">
        <h1 className="text-2xl font-bold text-neural-secondary mb-4">Error</h1>
        <p className="text-gray-300 mb-6">{error}</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neural-dark text-white">
      <Head>
        <title>Explore Models | NeuralPaper.ai</title>
        <meta name="description" content="Explore annotated neural network models" />
      </Head>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 text-neural-secondary">Explore Models</h1>
          <p className="text-xl text-gray-300">
            Discover annotated neural network architectures with interactive visualizations
          </p>
        </div>

        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 space-y-4 md:space-y-0">
          <div className="flex flex-wrap gap-2">
            {categories.map(category => (
              <button
                key={category}
                className={`px-4 py-2 rounded-full text-sm ${
                  filter === category
                    ? 'bg-neural-secondary text-white'
                    : 'bg-neural-primary text-gray-300 hover:bg-opacity-90'
                }`}
                onClick={() => setFilter(category)}
              >
                {category === 'all' ? 'All Categories' : category}
              </button>
            ))}
          </div>

          <div className="w-full md:w-auto">
            <input
              type="text"
              placeholder="Search models..."
              className="w-full md:w-64 px-4 py-2 bg-neural-primary text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-neural-secondary"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>

        {filteredModels.length === 0 ? (
          <div className="bg-neural-primary p-8 rounded-lg text-center">
            <h2 className="text-2xl font-semibold mb-4">No models found</h2>
            <p className="text-gray-300 mb-6">
              Try adjusting your search or filter criteria
            </p>
            <button
              className="px-4 py-2 bg-neural-secondary text-white rounded hover:bg-opacity-90"
              onClick={() => {
                setFilter('all');
                setSearchQuery('');
              }}
            >
              Reset Filters
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {filteredModels.map((model, index) => (
              <ModelCard
                key={model.id}
                model={model}
                index={index}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

interface ModelCardProps {
  model: ModelCard;
  index: number;
}

function ModelCard({ model, index }: ModelCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
    >
      <Link href={`/models/${model.id}`} className="block">
        <div className="bg-neural-primary rounded-lg overflow-hidden hover:shadow-lg transition-all">
          <div className="h-48 bg-gray-700 relative">
            {/* Replace with actual image */}
            <div className="absolute inset-0 flex items-center justify-center text-2xl font-bold">
              {model.name}
            </div>
          </div>
          <div className="p-6">
            <div className="flex justify-between items-start mb-2">
              <h3 className="text-xl font-semibold">{model.name}</h3>
              <span className="text-xs px-2 py-1 bg-neural-dark rounded-full text-gray-300">
                {model.complexity}
              </span>
            </div>
            <p className="text-gray-300 mb-4">{model.description}</p>
            <div className="flex justify-between items-center">
              <span className="text-sm text-neural-secondary">{model.category}</span>
              <span className="text-sm text-gray-400">View Details →</span>
            </div>
          </div>
        </div>
      </Link>
    </motion.div>
  );
}
