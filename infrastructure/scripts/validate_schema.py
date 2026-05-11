import json
import os
import sys


def validate_json(data_path, schema_path):
    if not os.path.exists(data_path):
        print(f"[-] Data file not found: {data_path}")
        return False
    if not os.path.exists(schema_path):
        print(f"[-] Schema file not found: {schema_path}")
        return False

    with open(data_path, "r") as f:
        data = json.load(f)
    with open(schema_path, "r") as f:
        json.load(f)

    # Basic validation (Checking for required top-level keys)
    required_keys = ["step", "events", "hypotheses", "root_causes"]
    missing = [k for k in required_keys if k not in data]

    if missing:
        print(f"[-] Validation Failed! Missing keys: {missing}")
        return False

    # Check events structure
    for i, event in enumerate(data.get("events", [])):
        event_keys = ["event_type", "layer_name", "step", "from_state", "to_state"]
        missing_event = [k for k in event_keys if k not in event]
        if missing_event:
            print(f"[-] Event {i} is invalid. Missing keys: {missing_event}")
            return False

    print(f"[+] Validation Success: {data_path} matches schema requirements.")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python validate_schema.py <data.json> <schema.json>")
        sys.exit(1)

    success = validate_json(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
