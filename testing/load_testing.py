import requests
import concurrent.futures
import time

# Define the URL for the API endpoint
url = 'http://127.0.0.1:5000/get_response'

# Sample payload for the requests
payload = {"question": "What is semester fee"}

# Function to send a single API request
def send_request(session, url, payload):
    response = session.post(url, json=payload)
    return response.json()

# Function to perform load testing
def load_test(url, payload, num_requests):
    with requests.Session() as session:
        start_time = time.time()
        
        # Use ThreadPoolExecutor to send requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(send_request, session, url, payload) for _ in range(num_requests)]
            
            responses = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    print(f"Request generated an exception: {e}")

        end_time = time.time()
        
    total_time = end_time - start_time
    average_time = total_time / num_requests if num_requests > 0 else 0

    return responses, total_time, average_time

if __name__ == "__main__":
    num_requests = 500# Number of concurrent requests to send

    print(f"Sending {num_requests} concurrent requests to {url}")
    responses, total_time, average_time = load_test(url, payload, num_requests)

    print(f"\nTotal time taken for {num_requests} requests: {total_time:.2f} seconds")
    print(f"Average time per request: {average_time:.2f} seconds")

    # Print the first few responses for verification
    for i, response in enumerate(responses[:5]):
        print(f"Response {i+1}: {response}")

    # Print the total number of responses received
    print(f"\nTotal responses received: {len(responses)}")
