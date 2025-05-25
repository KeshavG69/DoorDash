import pytest
import app

# Sample unit tests for app.py backend logic

def test_handle_user_input_valid(mocker):
    mocker.patch('app.call_main_pipeline', return_value="response")
    result = app.handle_user_input("hello")
    assert result == "response"

def test_handle_user_input_empty(mocker):
    result = app.handle_user_input("")
    assert result == ""  # or some error/fallback behavior

