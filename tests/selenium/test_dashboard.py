from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

def test_dashboard_ui():
    """Automates dashboard UI testing using Selenium."""
    driver = webdriver.Chrome()  # Requires chromedriver installed

    # Open NeuralDbg dashboard
    driver.get("http://localhost:8050")
    time.sleep(2)  # Wait for UI to load

    # Check if title is correct
    assert "NeuralDbg: Real-Time Execution Monitoring" in driver.title

    # Test interactive elements (e.g., clicking a button)
    step_debug_button = driver.find_element(By.ID, "step_debug_button")
    step_debug_button.click()
    time.sleep(1)  # Wait for action

    # Ensure debug message appears
    message = driver.find_element(By.ID, "step_debug_output").text
    assert "Paused. Check terminal for tensor inspection." in message

    driver.quit()
