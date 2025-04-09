import '../styles/globals.css';
import '../styles/monaco-custom.css';
import type { AppProps } from 'next/app';
import Head from 'next/head';
import { useEffect } from 'react';

// Import Monaco loader
import monacoLoader from '../lib/monaco';

function MyApp({ Component, pageProps }: AppProps) {
  // Initialize Monaco Editor
  useEffect(() => {
    // Pre-load Monaco Editor
    monacoLoader.init().then(monaco => {
      // Configure Monaco Editor
      monaco.editor.defineTheme('neural-dark', {
        base: 'vs-dark',
        inherit: true,
        rules: [
          { token: 'comment', foreground: '6A9955' },
          { token: 'keyword', foreground: 'C586C0' },
          { token: 'string', foreground: 'CE9178' },
        ],
        colors: {
          'editor.background': '#1a1a2e',
          'editor.foreground': '#d4d4d4',
          'editorCursor.foreground': '#e94560',
          'editor.lineHighlightBackground': '#2a2a3e',
          'editorLineNumber.foreground': '#6e6e8a',
          'editor.selectionBackground': '#3a3a5e',
          'editor.inactiveSelectionBackground': '#3a3a5e80',
        },
      });
    });
  }, []);

  return (
    <>
      <Head>
        <title>NeuralPaper.ai - Interactive Neural Network Models</title>
        <meta name="description" content="Explore annotated neural network architectures with interactive visualizations" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Component {...pageProps} />
    </>
  );
}

export default MyApp;
