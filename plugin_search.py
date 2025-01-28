import requests
import os

GITHUB_API_TOKEN = os.getenv('GITHUB_API_TOKEN', 'your_github_api_token')

def fetch_repo_details(full_name, headers):
    url = f"https://api.github.com/repos/{full_name}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Unable to fetch repo details for {full_name} (Status code: {response.status_code})")
        return None

def fetch_github_code_search_results(query):
    # Define the search URL and parameters
    url = "https://api.github.com/search/code"
    params = {
        'q': query,
        'per_page': 100
    }

    # Define the headers with authorization
    headers = {
        'Accept': 'application/vnd.github+json',
        'Authorization': f'Bearer {GITHUB_API_TOKEN}'
    }

    # Use a dictionary to track unique repositories
    unique_repositories = {}

    while url:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            for item in data.get('items', []):
                full_name = item['repository']['full_name']
                # Only process if we haven't seen this repository before
                if full_name not in unique_repositories:
                    repo_info = {
                        'url': item['repository']['html_url'],
                        'description': item['repository']['description'],
                        'full_name': full_name
                    }
                    # Fetch additional repo details
                    print(f"Fetch details for {repo_info['full_name']}")
                    repo_details = fetch_repo_details(repo_info['full_name'], headers)
                    if repo_details:
                        repo_info['updated_at'] = repo_details['updated_at']
                        repo_info['stars'] = repo_details['stargazers_count']
                        unique_repositories[full_name] = repo_info

            link_header = response.headers.get('Link')
            if link_header:
                links = link_header.split(',')
                next_url = None
                for link in links:
                    if 'rel="next"' in link:
                        next_url = link.split(';')[0].strip()[1:-1]  # Get the URL without < and >
                url = next_url  # Set the next URL for the next request
            else:
                url = None  # No more pages

        else:
            print(f"Error: Unable to fetch data from GitHub API (Status code: {response.status_code})")
            break

    # Convert the dictionary values back to a list
    return list(unique_repositories.values())

if __name__ == "__main__":
    query = '@llm.hookimpl'
    repositories = fetch_github_code_search_results(query)

    sorted_repositories = sorted(repositories, key=lambda x: x['updated_at'], reverse=True)
    
    markdown_content = "| Repository | Description | Stars | Last Updated | URL |\n"
    markdown_content += "|------------|-------------|-------|--------------|-----|\n"
    
    for repo_info in sorted_repositories:
        # Clean up description - replace None with empty string and escape any pipes
        description = (repo_info['description'] or "").replace("|", "\\|")
        markdown_content += f"| {repo_info['full_name']} | {description} | {repo_info['stars']} | {repo_info['updated_at'].split('T')[0]} | {repo_info['url']} |\n"    

    # Write to file
    with open('readme.md', 'w', encoding='utf-8') as f:
        f.write("# Unofficial LLM Plugin directory\n")
        f.write("This is auto-generated by querying Github. See [LLM Plugin directory](https://llm.datasette.io/en/stable/plugins/directory.html#plugin-directory) for the official curated list.\n\n")
        f.write(markdown_content)
    
    print("Results have been written to readme.md")