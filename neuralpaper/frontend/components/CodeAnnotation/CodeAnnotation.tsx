import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import ReactMarkdown from 'react-markdown';
import { motion } from 'framer-motion';
import { FiArrowRight, FiCode, FiInfo, FiBookOpen } from 'react-icons/fi';

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

  // Render line numbers with highlighting
  const renderLineNumbers = () => {
    return (
      <div className="line-numbers select-none pr-4 text-right text-gray-500 border-r border-gray-800">
        {codeLines.map((_, i) => {
          const lineNumber = i + 1;
          const isHighlighted = getLineHighlightStatus(lineNumber);
          const intensity = getHighlightIntensity(lineNumber);

          return (
            <div
              key={i}
              className={`line-number py-1 px-2 transition-colors duration-200 ${
                isHighlighted ? 'text-white font-medium' : ''
              }`}
              style={{
                backgroundColor: isHighlighted ? `rgba(233, 69, 96, ${intensity * 0.15})` : 'transparent',
              }}
              onMouseEnter={() => setHoveredLine(lineNumber)}
            >
              {lineNumber}
            </div>
          );
        })}
      </div>
    );
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

  // Render code with line highlighting
  const renderCode = () => {
    return (
      <div className="code-content relative flex-1 overflow-x-auto">
        <SyntaxHighlighter
          language={language}
          style={vscDarkPlus}
          showLineNumbers={false}
          wrapLines={true}
          lineProps={(lineNumber) => {
            const isHighlighted = getLineHighlightStatus(lineNumber);
            const intensity = getHighlightIntensity(lineNumber);

            return {
              style: {
                display: 'block',
                backgroundColor: isHighlighted ? `rgba(62, 68, 113, ${0.2 + intensity * 0.3})` : undefined,
                borderLeft: isHighlighted ? `3px solid rgba(233, 69, 96, ${0.7 + intensity * 0.3})` : undefined,
                paddingLeft: isHighlighted ? '16px' : '19px',
                transition: 'background-color 0.3s ease, border-left 0.3s ease',
              },
              onMouseEnter: () => setHoveredLine(lineNumber),
            };
          }}
          customStyle={{
            fontSize: '0.9rem',
            fontFamily: '"JetBrains Mono", "Fira Code", monospace',
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    );
  };

  // Scroll to active section
  const codeRef = useRef<HTMLDivElement>(null);
  const annotationRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (activeSection && codeRef.current && annotationRef.current) {
      const section = getActiveSection();
      if (section) {
        // Scroll code panel to active section
        const lineElements = codeRef.current.querySelectorAll('.line-number');
        if (lineElements.length >= section.lineStart) {
          lineElements[section.lineStart - 1].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        // Scroll annotation panel to top
        annotationRef.current.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }
  }, [activeSection]);

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

      <div className="flex flex-col lg:flex-row">
        {/* Code panel */}
        <div className="code-panel w-full lg:w-1/2 overflow-auto bg-[#1E1E2E] text-white border-r border-gray-800" ref={codeRef}>
          <div className="code-container flex min-h-[500px]">
            {renderLineNumbers()}
            {renderCode()}
          </div>
        </div>

        {/* Annotation panel */}
        <div className="annotation-panel w-full lg:w-1/2 bg-neural-primary p-6 overflow-auto min-h-[500px]" ref={annotationRef}>
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
            onClick={() => setActiveSection(section.id)}
            variants={itemVariants}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="w-5 h-5 rounded-full bg-neural-dark flex items-center justify-center mr-2 text-xs">
              {index + 1}
            </span>
            {`Lines ${section.lineStart}-${section.lineEnd}`}
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
};

export default CodeAnnotation;
