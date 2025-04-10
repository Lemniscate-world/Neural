# twitter_bot.py
import tweepy
import os
import re

auth = tweepy.OAuthHandler(os.environ["TWITTER_API_KEY"], os.environ["TWITTER_API_SECRET"])
auth.set_access_token(os.environ["TWITTER_ACCESS_TOKEN"], os.environ["TWITTER_ACCESS_TOKEN_SECRET"])
api = tweepy.API(auth)

def parse_changelog(version):
    """Extract changelog entries for a specific version from CHANGELOG.md"""
    with open("CHANGELOG.md", "r") as f:
        changelog = f.read()

    # Split changelog into version sections
    sections = re.split(r"## \[(\d+\.\d+\.\d+)\]", changelog)
    for i in range(1, len(sections), 2):
        if sections[i] == version:
            content = sections[i+1].strip()
            # Extract list items (bullet points)
            changes = re.findall(r"- (.*?)\n", content)
            return changes[:5]  # Return first 5 changes to avoid truncation
    raise ValueError(f"Version {version} not found in CHANGELOG.md")

def format_tweet(version, changes):
    """Create tweet text with version, changes, and truncate if needed"""
    max_length = 250
    change_text = "\n".join(f"• {change}" for change in changes)
    base_text = f"🚀 Neural-dsl v{version} released!\n{change_text}\n\n#MachineLearning #Python\nGitHub: https://github.com/Lemniscate-world/Neuralreleases/tag/v{version}"

    if len(base_text) > max_length:
        base_text = base_text[:max_length-3] + "..."
    return base_text

def post_release(version):
    try:
        changes = parse_changelog(version)
        tweet_text = format_tweet(version, changes)
        api.update_status(tweet_text)
        print("Tweet posted successfully!")
    except Exception as e:
        print(f"Error posting tweet: {str(e)}")

# Example usage: post_release("1.2.3")
