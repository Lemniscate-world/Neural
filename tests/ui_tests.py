from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def test_dashboard_ui():
    """Automates UI testing using Selenium."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Chrome(options=options)

    # Open NeuralDbg dashboard
    driver.get("http://localhost:8050")
    time.sleep(2)  # Wait for UI to load

    # Check if title is correct
    assert "NeuralDbg: Real-Time Execution Monitoring" in driver.title

    # Test Step Debugging Button
    try:
        step_debug_button = driver.find_element(By.ID, "step_debug_button")
        step_debug_button.click()
        time.sleep(1)
        
        message = driver.find_element(By.ID, "step_debug_output").text
        assert "Paused. Check terminal for tensor inspection." in message
    except Exception:
        driver.save_screenshot("failed_test.png")  # Take screenshot if test fails
        raise

    driver.quit()
