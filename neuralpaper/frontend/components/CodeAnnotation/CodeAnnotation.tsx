import React, { useState, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import ReactMarkdown from 'react-markdown';

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
      <div className="line-numbers select-none pr-4 text-right text-gray-500 border-r border-gray-700">
        {codeLines.map((_, i) => {
          const lineNumber = i + 1;
          const isHighlighted = getLineHighlightStatus(lineNumber);
          
          return (
            <div
              key={i}
              className={`line-number ${
                isHighlighted ? 'text-white font-medium' : ''
              }`}
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
            return {
              style: {
                display: 'block',
                backgroundColor: isHighlighted ? 'rgba(62, 68, 113, 0.3)' : undefined,
                borderLeft: isHighlighted ? '3px solid #e94560' : undefined,
                paddingLeft: isHighlighted ? '16px' : '19px',
              },
              onMouseEnter: () => setHoveredLine(lineNumber),
            };
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    );
  };

  return (
    <div className="code-annotation-container bg-neural-dark rounded-lg overflow-hidden">
      {title && (
        <div className="code-title bg-neural-primary px-4 py-2 text-white font-medium">
          {title}
        </div>
      )}
      
      <div className="flex flex-col md:flex-row">
        {/* Code panel */}
        <div className="code-panel w-full md:w-1/2 overflow-auto bg-[#1E1E2E] text-white">
          <div className="code-container flex">
            {renderLineNumbers()}
            {renderCode()}
          </div>
        </div>
        
        {/* Annotation panel */}
        <div className="annotation-panel w-full md:w-1/2 bg-neural-primary p-6 overflow-auto">
          {getActiveSection() ? (
            <div className="annotation-content">
              <h3 className="text-xl font-semibold mb-4 text-neural-secondary">
                {`Lines ${getActiveSection()?.lineStart}-${getActiveSection()?.lineEnd}`}
              </h3>
              <div className="prose prose-invert max-w-none">
                <ReactMarkdown>{getActiveSection()?.annotation || ''}</ReactMarkdown>
              </div>
            </div>
          ) : (
            <div className="text-gray-400 italic">
              Hover over code to see annotations
            </div>
          )}
        </div>
      </div>
      
      {/* Section navigation */}
      <div className="section-nav bg-neural-primary border-t border-gray-700 px-4 py-2 flex overflow-x-auto">
        {sections.map((section) => (
          <button
            key={section.id}
            className={`px-3 py-1 mr-2 text-sm rounded ${
              activeSection === section.id
                ? 'bg-neural-secondary text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
            onClick={() => setActiveSection(section.id)}
          >
            {`Lines ${section.lineStart}-${section.lineEnd}`}
          </button>
        ))}
      </div>
    </div>
  );
};

export default CodeAnnotation;
