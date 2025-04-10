import os
import sys
import pytest
from lark import exceptions
from lark.exceptions import VisitError

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural.parser.parser import ModelTransformer, create_parser, DSLValidationError

class TestResearchParsing:
    @pytest.fixture
    def research_parser(self):
        return create_parser('research')

    @pytest.fixture
    def transformer(self):
        return ModelTransformer()

    @pytest.mark.parametrize(
        "research_string, expected_name, expected_metrics, expected_references, test_id",
        [
            (
                """
                research ResearchStudy {
                    metrics {
                        accuracy: 0.95
                        loss: 0.05
                    }
                    references {
                        paper: "Paper Title 1"
                        paper: "Another Great Paper"
                    }
                }
                """,
                "ResearchStudy", {'accuracy': 0.95, 'loss': 0.05}, ["Paper Title 1", "Another Great Paper"],
                "complete-research"
            ),
            (
                """
                research {
                    metrics {
                        precision: 0.8
                        recall: 0.9
                    }
                }
                """,
                None, {'precision': 0.8, 'recall': 0.9}, [],
                "no-name-no-ref"
            ),
            (
                """
                research InvalidMetrics {
                    metrics {
                        accuracy: "high"
                    }
                }
                """,
                None, None, None,
                "invalid-metrics"
            ),
        ],
        ids=["complete-research", "no-name-no-ref", "invalid-metrics"]
    )
    def test_research_parsing(self, research_parser, transformer, research_string, expected_name, expected_metrics, expected_references, test_id):
        if expected_metrics is None:
            with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError)):
                tree = research_parser.parse(research_string)
                transformer.transform(tree)
        else:
            tree = research_parser.parse(research_string)
            result = transformer.transform(tree)
            assert result['type'] == 'Research'
            assert result['name'] == expected_name
            assert result.get('params', {}).get('metrics', {}) == expected_metrics
            assert result.get('params', {}).get('references', []) == expected_references
