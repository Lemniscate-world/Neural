import React, { useState, useEffect, useRef } from 'react';
import Editor from '@monaco-editor/react';
import ReactMarkdown from 'react-markdown';
import { motion } from 'framer-motion';
import { FiArrowRight, FiCode, FiInfo, FiBookOpen } from 'react-icons/fi';
import * as monaco from 'monaco-editor';

interface CodeSection {
  id: string;
  code: string;
  annotation: string;
  lineStart: number;
  lineEnd: number;
}

interface CodeAnnotationProps {
  code: string;
  language: string;
  sections: CodeSection[];
  title?: string;
}

const CodeAnnotation: React.FC<CodeAnnotationProps> = ({
  code,
  language,
  sections,
  title,
}) => {
  const [activeSection, setActiveSection] = useState<string | null>(null);
  const [codeLines, setCodeLines] = useState<string[]>([]);
  const [hoveredLine, setHoveredLine] = useState<number | null>(null);

  useEffect(() => {
    // Split code into lines
    setCodeLines(code.split('\n'));

    // Set first section as active by default
    if (sections.length > 0 && !activeSection) {
      setActiveSection(sections[0].id);
    }
  }, [code, sections, activeSection]);

  // Find the section that contains the hovered line
  useEffect(() => {
    if (hoveredLine !== null) {
      const section = sections.find(
        (s) => hoveredLine >= s.lineStart && hoveredLine <= s.lineEnd
      );
      if (section) {
        setActiveSection(section.id);
      }
    }
  }, [hoveredLine, sections]);

  // Get the active section object
  const getActiveSection = () => {
    return sections.find((s) => s.id === activeSection) || null;
  };

  // Create editor decorations for highlighted sections
  const createEditorDecorations = (editor: monaco.editor.IStandaloneCodeEditor | null) => {
    if (!editor) return;

    // Clear existing decorations
    const model = editor.getModel();
    if (!model) return;

    // Create decorations for each section
    const decorations = sections.map(section => {
      const isActive = activeSection === section.id;
      return {
        range: new monaco.Range(
          section.lineStart,
          1,
          section.lineEnd,
          model.getLineMaxColumn(section.lineEnd)
        ),
        options: {
          isWholeLine: true,
          className: isActive ? 'active-section-highlight' : 'section-highlight',
          linesDecorationsClassName: isActive ? 'active-section-decoration' : 'section-decoration',
          inlineClassName: isActive ? 'active-section-inline' : 'section-inline',
          marginClassName: isActive ? 'active-section-margin' : 'section-margin'
        }
      };
    });

    // Apply decorations
    editor.deltaDecorations([], decorations);
  };

  // Determine if a line should be highlighted
  const getLineHighlightStatus = (lineNumber: number): boolean => {
    const active = getActiveSection();
    if (!active) return false;
    return lineNumber >= active.lineStart && lineNumber <= active.lineEnd;
  };

  // Get highlight intensity (for gradient effect)
  const getHighlightIntensity = (lineNumber: number): number => {
    const active = getActiveSection();
    if (!active) return 0;
    if (lineNumber < active.lineStart || lineNumber > active.lineEnd) return 0;

    // Create a gradient effect with highest intensity in the middle
    const totalLines = active.lineEnd - active.lineStart + 1;
    const position = lineNumber - active.lineStart;
    const normalizedPos = position / totalLines;

    // Bell curve intensity (highest in the middle)
    return Math.sin(normalizedPos * Math.PI) * 0.5 + 0.5;
  };

  // Handle editor mount
  const handleEditorDidMount = (editor: monaco.editor.IStandaloneCodeEditor) => {
    // Store editor reference
    editorRef.current = editor;

    // Add custom CSS classes
    const styleElement = document.createElement('style');
    styleElement.textContent = `
      .section-highlight { background-color: rgba(62, 68, 113, 0.1); }
      .active-section-highlight { background-color: rgba(62, 68, 113, 0.2); }
      .section-decoration { border-left: 2px solid rgba(233, 69, 96, 0.5); margin-left: 3px; }
      .active-section-decoration { border-left: 3px solid rgba(233, 69, 96, 0.8); margin-left: 2px; }
      .monaco-editor .line-numbers { font-family: 'JetBrains Mono', 'Fira Code', monospace !important; }
    `;
    document.head.appendChild(styleElement);

    // Create decorations
    createEditorDecorations(editor);

    // Add mouse move event listener to detect hovered lines
    editor.onMouseMove((e) => {
      if (e.target.position) {
        const lineNumber = e.target.position.lineNumber;
        setHoveredLine(lineNumber);
      }
    });

    // Add click event listener to navigate to section
    editor.onMouseDown((e) => {
      if (e.target.position) {
        const lineNumber = e.target.position.lineNumber;
        const section = sections.find(
          (s) => lineNumber >= s.lineStart && lineNumber <= s.lineEnd
        );
        if (section) {
          setActiveSection(section.id);
        }
      }
    });
  };

  // Editor reference
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);

  // Scroll to active section
  const codeRef = useRef<HTMLDivElement>(null);
  const annotationRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (activeSection && editorRef.current && annotationRef.current) {
      const section = getActiveSection();
      if (section) {
        // Scroll code panel to active section
        editorRef.current.revealLineInCenter(section.lineStart);

        // Update decorations
        createEditorDecorations(editorRef.current);

        // Scroll annotation panel to top
        annotationRef.current.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }
  }, [activeSection]);

  // Update decorations when sections or active section changes
  useEffect(() => {
    if (editorRef.current) {
      createEditorDecorations(editorRef.current);
    }
  }, [sections, activeSection]);

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1 }
  };

  return (
    <motion.div
      className="code-annotation-container bg-neural-dark rounded-xl overflow-hidden border border-gray-800 shadow-2xl"
      initial="hidden"
      animate="visible"
      variants={containerVariants}
    >
      {title && (
        <div className="code-title bg-neural-primary px-6 py-4 text-white font-medium flex items-center border-b border-gray-800">
          <div className="w-8 h-8 rounded-full bg-neural-secondary bg-opacity-20 flex items-center justify-center mr-3">
            <FiCode className="text-neural-secondary" />
          </div>
          <span className="text-xl">{title}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-0 h-full">
        {/* Code panel */}
        <div className="code-panel h-[500px] bg-[#1E1E2E] text-white border-r border-gray-800">
          <Editor
            height="100%"
            defaultLanguage={language}
            value={code}
            theme="vs-dark"
            options={{
              readOnly: true,
              minimap: { enabled: false },
              fontSize: 14,
              wordWrap: 'on',
              scrollBeyondLastLine: false,
              lineNumbers: 'on',
              glyphMargin: true,
              folding: true,
              lineDecorationsWidth: 10,
              renderLineHighlight: 'none',
              automaticLayout: true,
              fontFamily: '"JetBrains Mono", "Fira Code", monospace',
              fontLigatures: true
            }}
            onMount={handleEditorDidMount}
          />
        </div>

        {/* Annotation panel */}
        <div className="annotation-panel bg-neural-primary p-6 overflow-auto h-[500px]" ref={annotationRef}>
          {getActiveSection() ? (
            <motion.div
              className="annotation-content"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              key={activeSection} // Force re-animation when section changes
            >
              <div className="flex items-center mb-4">
                <div className="w-8 h-8 rounded-full bg-neural-secondary bg-opacity-20 flex items-center justify-center mr-3">
                  <FiBookOpen className="text-neural-secondary" />
                </div>
                <h3 className="text-xl font-semibold text-neural-secondary">
                  {`Lines ${getActiveSection()?.lineStart}-${getActiveSection()?.lineEnd}`}
                </h3>
              </div>

              <div className="prose prose-invert max-w-none bg-neural-dark bg-opacity-30 p-4 rounded-lg border border-gray-800 shadow-inner">
                <ReactMarkdown>{getActiveSection()?.annotation || ''}</ReactMarkdown>
              </div>

              <div className="mt-4 text-sm text-gray-400 flex items-center">
                <FiInfo className="mr-2" />
                <span>Hover over code to navigate between sections</span>
              </div>
            </motion.div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-gray-400">
              <FiArrowRight className="text-4xl mb-4 text-neural-secondary animate-pulse" />
              <p className="text-center">Hover over code to see annotations</p>
            </div>
          )}
        </div>
      </div>

      {/* Section navigation */}
      <div className="section-nav bg-neural-primary border-t border-gray-800 px-6 py-3 flex overflow-x-auto">
        {sections.map((section, index) => (
          <motion.button
            key={section.id}
            className={`px-4 py-2 mr-3 text-sm rounded-lg transition-all duration-300 flex items-center ${
              activeSection === section.id
                ? 'bg-neural-secondary text-white shadow-lg'
                : 'bg-neural-dark bg-opacity-50 text-gray-300 hover:bg-opacity-70'
            }`}
            onClick={() => {
              setActiveSection(section.id);
              // Scroll to the section in the editor
              if (editorRef.current) {
                editorRef.current.revealLineInCenter(section.lineStart);
              }
            }}
            variants={itemVariants}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="w-5 h-5 rounded-full bg-neural-dark flex items-center justify-center mr-2 text-xs">
              {index + 1}
            </span>
            <span className="whitespace-nowrap">{`Lines ${section.lineStart}-${section.lineEnd}`}</span>
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
};

export default CodeAnnotation;
