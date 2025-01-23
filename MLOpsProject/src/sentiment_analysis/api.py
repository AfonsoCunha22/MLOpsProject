from google.cloud import storage


def download_data(bucket_name, source_blob_name, destination_file_name):
    """
    Downloads a blob from the specified Google Cloud Storage bucket.

    Args:
        bucket_name (str): The name of the bucket.
        source_blob_name (str): The name of the blob to download.
        destination_file_name (str): The local file path to save the downloaded blob.
    """
    # Initialize a storage client
    storage_client = storage.Client()

    # Get the bucket object
    bucket = storage_client.bucket(bucket_name)

    # Get the blob object
    blob = bucket.blob(source_blob_name)

    # Download the blob to the specified local file
    blob.download_to_filename(destination_file_name)

    # Print a confirmation message
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


if __name__ == "__main__":
    # Example usage
    bucket_name = "your-bucket-name"  # Replace with your bucket name
    source_blob_name = "path/to/your/datafile"  # Replace with the path to your blob
    destination_file_name = "data/raw/datafile"  # Replace with your desired local file path

    # Call the download function
    download_data(bucket_name, source_blob_name, destination_file_name)
