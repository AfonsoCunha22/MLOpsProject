from locust import HttpUser, between, task
import random

"""
running it online

locust -f tests/locustfile.py
"""


"""
Running it in the terminal
in this case below:
    - Simulates 10 users total.
    - Spawns them at 2 users/second.
    - Continues for 1 minute.
    - Targets your local FastAPI endpoint.

export MYENDPOINT=http://127.0.0.1:8000

locust -f tests/locustfile.py \
    --headless \
    --users 10 \
    --spawn-rate 2 \
    --run-time 1m \
    --host $MYENDPOINT
"""


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    # A short wait time keeps the user "busy"
    wait_time = between(1, 3)

    @task
    def root_endpoint(self):
        # Simple GET request to the root endpoint:
        self.client.get("/")

    @task(3)
    def predict_endpoint(self):
        # More frequent requests to the /predict endpoint
        text_options = [
            "I really like this product!",
            "This is terrible.",
            "I'm not sure how I feel about this.",
            "Absolutely amazing service!",
            "I will never use this again.",
        ]
        # Randomly pick a text to post
        text = random.choice(text_options)
        payload = {"text": text}

        # Send a POST request to /predict
        response = self.client.post("/predict/", json=payload)

        # Optional: you could check something about response status or data
        # But typically Locust is for load metrics, not functional tests
        if response.status_code != 200:
            print(f"Got unexpected status code: {response.status_code}")
