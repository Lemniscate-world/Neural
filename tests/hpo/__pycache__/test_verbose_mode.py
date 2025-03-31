import os
import sys
import pytest
from unittest.mock import patch
from click.testing import CliRunner

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.cli import cli

class TestVerboseMode:
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_verbose_mode(self, runner):
        """Test that verbose mode is properly configured."""
        # This test will verify that the verbose flag is properly passed to configure_logging
        with patch('neural.cli.configure_logging') as mock_configure_logging:
            # Run with verbose flag
            runner.invoke(cli, ['-v', 'version'])
            mock_configure_logging.assert_called_with(True)
            
            # Run without verbose flag
            runner.invoke(cli, ['version'])
            mock_configure_logging.assert_called_with(False)
