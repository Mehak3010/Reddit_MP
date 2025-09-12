import praw
import os

# Reddit API credentials (replace with your own)
# It's recommended to use environment variables for sensitive information
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'BotDetectionScript by /u/YourRedditUsername')

# Initialize PRAW
try:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    print("PRAW initialized successfully.")
except Exception as e:
    print(f"Error initializing PRAW: {e}")
    print("Please ensure REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT environment variables are set.")
    exit()

def get_user_data(username, limit=100):
    """Fetches a user's comment and submission history."""
    try:
        user = reddit.redditor(username)
        print(f"Fetching data for user: {username}")

        comments = []
        for comment in user.comments.new(limit=limit):
            comments.append({
                'id': comment.id,
                'subreddit': comment.subreddit.display_name,
                'body': comment.body,
                'score': comment.score,
                'created_utc': comment.created_utc
            })
        print(f"Fetched {len(comments)} comments.")

        submissions = []
        for submission in user.submissions.new(limit=limit):
            submissions.append({
                'id': submission.id,
                'subreddit': submission.subreddit.display_name,
                'title': submission.title,
                'selftext': submission.selftext,
                'score': submission.score,
                'created_utc': submission.created_utc
            })
        print(f"Fetched {len(submissions)} submissions.")

        return {'comments': comments, 'submissions': submissions}
    except Exception as e:
        print(f"Error fetching data for user {username}: {e}")
        return None

def get_subreddit_posts(subreddit_name, limit=100):
    """Fetches recent posts from a subreddit."""
    try:
        subreddit = reddit.subreddit(subreddit_name)
        print(f"Fetching posts from subreddit: {subreddit_name}")

        posts = []
        for submission in subreddit.new(limit=limit):
            posts.append({
                'id': submission.id,
                'title': submission.title,
                'selftext': submission.selftext,
                'author': submission.author.name if submission.author else '[deleted]',
                'score': submission.score,
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc
            })
        print(f"Fetched {len(posts)} posts.")
        return posts
    except Exception as e:
        print(f"Error fetching posts from subreddit {subreddit_name}: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    # To run this, you need to set the environment variables:
    # REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

    # Example: Get data for a specific user
    # user_data = get_user_data('spez')
    # if user_data:
    #     print("\n--- User Data for spez ---")
    #     print(f"Comments: {len(user_data['comments'])}")
    #     print(f"Submissions: {len(user_data['submissions'])}")

    # Example: Get posts from a specific subreddit
    # subreddit_posts = get_subreddit_posts('MachineLearning')
    # if subreddit_posts:
    #     print("\n--- Subreddit Posts for MachineLearning ---")
    #     print(f"Posts: {len(subreddit_posts)}")

    print("Script `reddit_data_collector.py` created. Please set your Reddit API credentials as environment variables and uncomment the example usage to test.")
    print("For more information on setting up PRAW and obtaining credentials, refer to the PRAW documentation.")