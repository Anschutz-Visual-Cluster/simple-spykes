import uuid


def save_json(json_data, filename):
        other_filename = f"quality_metrics_{str(uuid.uuid4())}"
        try:
            print(f"Saving metrics to file '{filename}'")
            fp = open(filename, "w")
            fp.write(json_data)
            fp.close()
        except Exception as e:
            print(f"Error saving metrics to specified file '{filename}'! Saving to file '{other_filename}' \nError: {str(e)}")
            fp = open(other_filename, "w")
            fp.write(json_data)
            fp.close()
