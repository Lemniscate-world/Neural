import { useState, useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';
import { motion } from 'framer-motion';

export default function Home() {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  return (
    <div className="min-h-screen bg-neural-dark text-white">
      <Head>
        <title>NeuralPaper.ai - Interactive Neural Network Models</title>
        <meta name="description" content="Explore annotated neural network architectures with interactive visualizations" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: isLoaded ? 1 : 0, y: isLoaded ? 0 : 20 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-5xl font-bold mb-6 text-neural-secondary">
            NeuralPaper<span className="text-blue-400">.ai</span>
          </h1>
          <p className="text-xl max-w-3xl mx-auto text-gray-300">
            Interactive, annotated neural network models with visualization and debugging tools.
            Built with Neural DSL and NeuralDbg.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          <FeatureCard 
            title="Annotated Models"
            description="Explore neural network architectures with detailed annotations explaining each component."
            icon="/icons/annotation.svg"
            delay={0.1}
            isLoaded={isLoaded}
          />
          <FeatureCard 
            title="Interactive Visualization"
            description="Visualize model architecture, shape propagation, and computation flow in real-time."
            icon="/icons/visualization.svg"
            delay={0.2}
            isLoaded={isLoaded}
          />
          <FeatureCard 
            title="Live Debugging"
            description="Debug models with NeuralDbg to analyze gradients, activations, and performance."
            icon="/icons/debug.svg"
            delay={0.3}
            isLoaded={isLoaded}
          />
          <FeatureCard 
            title="DSL Playground"
            description="Experiment with Neural DSL to create and modify models with instant feedback."
            icon="/icons/code.svg"
            delay={0.4}
            isLoaded={isLoaded}
          />
          <FeatureCard 
            title="Educational Resources"
            description="Learn about neural network concepts with interactive examples and tutorials."
            icon="/icons/education.svg"
            delay={0.5}
            isLoaded={isLoaded}
          />
          <FeatureCard 
            title="Research Integration"
            description="Connect models to research papers with citations and explanations."
            icon="/icons/research.svg"
            delay={0.6}
            isLoaded={isLoaded}
          />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: isLoaded ? 1 : 0, y: isLoaded ? 0 : 20 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          className="text-center"
        >
          <h2 className="text-3xl font-bold mb-8 text-neural-secondary">
            Explore Popular Models
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <ModelCard 
              title="ResNet"
              description="Deep residual networks with skip connections"
              image="/models/resnet.png"
              href="/models/resnet"
            />
            <ModelCard 
              title="Transformer"
              description="Attention-based sequence model for NLP tasks"
              image="/models/transformer.png"
              href="/models/transformer"
            />
            <ModelCard 
              title="MLP-Mixer"
              description="All-MLP architecture for image classification"
              image="/models/mlp-mixer.png"
              href="/models/mlp-mixer"
            />
          </div>
          <div className="mt-12">
            <Link href="/models" className="px-8 py-3 bg-neural-secondary text-white rounded-lg hover:bg-opacity-90 transition-all">
              View All Models
            </Link>
          </div>
        </motion.div>
      </main>

      <footer className="bg-neural-primary py-8 text-center text-gray-300">
        <div className="container mx-auto px-4">
          <p>Built with Neural DSL and NeuralDbg</p>
          <p className="mt-2">© {new Date().getFullYear()} NeuralPaper.ai</p>
        </div>
      </footer>
    </div>
  );
}

interface FeatureCardProps {
  title: string;
  description: string;
  icon: string;
  delay: number;
  isLoaded: boolean;
}

function FeatureCard({ title, description, icon, delay, isLoaded }: FeatureCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: isLoaded ? 1 : 0, y: isLoaded ? 0 : 20 }}
      transition={{ duration: 0.5, delay }}
      className="bg-neural-primary p-6 rounded-lg hover:shadow-lg transition-all"
    >
      <div className="w-12 h-12 mb-4 mx-auto">
        {/* Replace with actual icon */}
        <div className="w-12 h-12 bg-neural-secondary rounded-full flex items-center justify-center">
          <span className="text-xl font-bold">{title.charAt(0)}</span>
        </div>
      </div>
      <h3 className="text-xl font-semibold mb-2 text-center">{title}</h3>
      <p className="text-gray-300 text-center">{description}</p>
    </motion.div>
  );
}

interface ModelCardProps {
  title: string;
  description: string;
  image: string;
  href: string;
}

function ModelCard({ title, description, image, href }: ModelCardProps) {
  return (
    <Link href={href} className="block">
      <div className="bg-neural-primary rounded-lg overflow-hidden hover:shadow-lg transition-all">
        <div className="h-48 bg-gray-700 relative">
          {/* Replace with actual image */}
          <div className="absolute inset-0 flex items-center justify-center text-2xl font-bold">
            {title}
          </div>
        </div>
        <div className="p-4">
          <h3 className="text-xl font-semibold mb-2">{title}</h3>
          <p className="text-gray-300">{description}</p>
        </div>
      </div>
    </Link>
  );
}
