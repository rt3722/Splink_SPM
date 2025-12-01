import os
import requests
from datetime import datetime
import time


def get_github_data(owner, repo, data_type, output_file):
    """Fetch issues or discussions from GitHub API and write to file"""
    print(f"\nFetching {data_type} from {owner}/{repo}...")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN", "")
    
    if not token:
        print("Error: No GitHub token found. Set GITHUB_TOKEN or GH_TOKEN environment variable.")
        return
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {token}",
    }

    count = 0
    with open(output_file, "a", encoding="utf-8") as f:
        if data_type == "issues":
            url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=all&per_page=100"
            page = 1

            while True:
                print(f"Fetching page {page}...", end="\r")
                response = requests.get(f"{url}&page={page}", headers=headers)
                if response.status_code != 200:
                    print(f"Error: {response.status_code}, {response.text}")
                    break

                items = response.json()
                if not items:
                    break

                for item in items:
                    comments_url = item["comments_url"]
                    comments_response = requests.get(comments_url, headers=headers)
                    if comments_response.status_code == 200:
                        item["comments"] = comments_response.json()
                    else:
                        item["comments"] = []

                    f.write(format_thread(item, is_issue=True))
                    count += 1
                    time.sleep(0.01)

                page += 1
                time.sleep(0.01)

        else:  # For discussions
            url = "https://api.github.com/graphql"
            query = """
            query($owner: String!, $repo: String!, $cursor: String) {
                repository(owner: $owner, name: $repo) {
                    discussions(first: 100, after: $cursor) {
                        nodes {
                            number
                            title
                            body
                            createdAt
                            repository {
                                name
                                owner {
                                    login
                                }
                            }
                            comments(first: 100) {
                                nodes {
                                    body
                                    createdAt
                                }
                            }
                        }
                        pageInfo {
                            endCursor
                            hasNextPage
                        }
                    }
                }
            }
            """
            variables = {"owner": owner, "repo": repo, "cursor": None}
            cursor = None

            while True:
                if cursor:
                    variables["cursor"] = cursor
                response = requests.post(
                    url, headers=headers, json={"query": query, "variables": variables}
                )
                if response.status_code != 200:
                    print(f"Error: {response.status_code}, {response.text}")
                    break

                data = response.json()
                if "errors" in data:
                    print(data["errors"])
                    break

                page_data = data["data"]["repository"]["discussions"]
                items = page_data["nodes"]
                if not items:
                    break

                for item in items:
                    f.write(format_thread(item, is_issue=False))
                    count += 1

                cursor = page_data["pageInfo"]["endCursor"]
                if not page_data["pageInfo"]["hasNextPage"]:
                    break
                time.sleep(0.01)

    print(f"\nFound {count} {data_type}")


def format_thread(item, is_issue=True):
    """Format a single issue or discussion thread"""
    created_at = datetime.fromisoformat(
        item["created_at" if is_issue else "createdAt"].replace("Z", "+00:00")
    )

    # Get URL based on type
    if is_issue:
        url = item["html_url"]
    else:
        # For discussions, we need to construct the URL since it's not provided in the GraphQL response
        url = f"https://github.com/{item['repository']['owner']['login']}/{item['repository']['name']}/discussions/{item['number']}"

    output = [
        f"{'ISSUE' if is_issue else 'DISCUSSION'}: {item['title']}",
        f"URL: {url}",
        f"Created: {created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Body:\n{item['body']}\n",
        "Comments:\n",
    ]

    if is_issue and "comments" in item:
        for comment in item.get("comments", []):
            # Skip comments from github-actions[bot]
            if comment["user"]["login"] == "github-actions[bot]":
                continue

            comment_time = datetime.fromisoformat(
                comment["created_at"].replace("Z", "+00:00")
            )
            output.append(
                f"--- Comment by {comment['user']['login']} at {comment_time.strftime('%Y-%m-%d %H:%M:%S UTC')} ---"
            )
            output.append(f"{comment['body']}\n")

    if not is_issue and "comments" in item and item["comments"]["nodes"]:
        for comment in item["comments"]["nodes"]:
            comment_time = datetime.fromisoformat(
                comment["createdAt"].replace("Z", "+00:00")
            )
            output.append(
                f"--- Comment at {comment_time.strftime('%Y-%m-%d %H:%M:%S UTC')} ---"
            )
            output.append(f"{comment['body']}\n")

    return "\n".join(output) + "\n" + "-" * 80 + "\n"


def main():
    owner = "moj-analytical-services"
    repo = "splink"
    output_file = "splink_knowledge_base.txt"

    print("Starting GitHub data collection...")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("SPLINK GITHUB ISSUES AND DISCUSSIONS KNOWLEDGE BASE\n")
        f.write("=" * 50 + "\n\n")
        f.write("ISSUES\n")
        f.write("-" * 50 + "\n\n")

    get_github_data(owner, repo, "issues", output_file)

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\nDISCUSSIONS\n")
        f.write("-" * 50 + "\n\n")

    get_github_data(owner, repo, "discussions", output_file)

    print("Done!")


if __name__ == "__main__":
    main()
