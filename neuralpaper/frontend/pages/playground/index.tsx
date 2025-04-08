import { useState } from 'react';
import Head from 'next/head';
import dynamic from 'next/dynamic';

// Import DSLPlayground dynamically to avoid SSR issues with Monaco editor
const DSLPlayground = dynamic(
  () => import('../../components/DSLPlayground/DSLPlayground'),
  { ssr: false }
);

export default function PlaygroundPage() {
  return (
    <div className="min-h-screen bg-neural-dark text-white">
      <Head>
        <title>Neural DSL Playground | NeuralPaper.ai</title>
        <meta name="description" content="Interactive playground for experimenting with Neural DSL" />
      </Head>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 text-neural-secondary">Neural DSL Playground</h1>
          <p className="text-xl text-gray-300">
            Experiment with Neural DSL to create and modify models with instant feedback.
          </p>
        </div>

        <div className="bg-neural-primary p-4 rounded-lg mb-8">
          <h2 className="text-xl font-semibold mb-2">Getting Started</h2>
          <ul className="list-disc list-inside text-gray-300 space-y-2">
            <li>Edit the Neural DSL code in the editor</li>
            <li>Click <strong>Run</strong> to visualize the model architecture</li>
            <li>Click <strong>Generate Code</strong> to see the equivalent TensorFlow or PyTorch code</li>
            <li>Click <strong>Debug</strong> to launch NeuralDbg for real-time execution monitoring</li>
          </ul>
        </div>

        <div className="h-[800px]">
          <DSLPlayground height="700px" />
        </div>
      </main>
    </div>
  );
}
