#!/usr/bin/env python3
# Test.py - Simple Python script for Electron integration

import sys
import json
import platform


def get_system_info():
    """Get basic system information"""
    return {
        "python_version": sys.version,
        "platform": platform.system(),
        "platform_version": platform.version(),
        "platform_release": platform.release(),
        "architecture": platform.machine()
    }


def process_data(data):
    """Process input data and return a result"""
    try:
        if isinstance(data, dict):
            # Echo back the data with a confirmation message
            result = {
                "status": "success",
                "message": "Data received successfully",
                "input": data,
                "processed": True,
                "system_info": get_system_info()
            }
            return result
        else:
            return {
                "status": "error",
                "message": "Input must be a JSON object",
                "system_info": get_system_info()
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "system_info": get_system_info()
        }


if __name__ == "__main__":
    # Check if any arguments were passed
    if len(sys.argv) > 1:
        try:
            # Try to parse input as JSON
            input_data = json.loads(sys.argv[1])
            result = process_data(input_data)
        except json.JSONDecodeError:
            # If not JSON, treat as simple string
            result = process_data({"text": sys.argv[1]})
    else:
        # No arguments, return system info
        result = {"status": "success", "system_info": get_system_info()}

    # Print the result as JSON for the Electron app to parse
    print(json.dumps(result, indent=2))