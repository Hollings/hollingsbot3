import os
import json
import urllib.request


def main():
    token = os.environ['DISCORD_TOKEN']
    channel_id = os.environ['DEBUG_CHANNEL_ID']

    pr_number = os.environ.get('PR_NUMBER', '')
    pr_title = os.environ.get('PR_TITLE', '')
    pr_url = os.environ.get('PR_URL', '')
    merged_by = os.environ.get('MERGED_BY', '')
    additions = os.environ.get('ADDITIONS', '')
    deletions = os.environ.get('DELETIONS', '')
    changed_files = os.environ.get('CHANGED_FILES', '')

    msg = (
        f"PR #{pr_number} merged by {merged_by}\n"
        f"{pr_title}\n{pr_url}\n"
        f"+{additions} -{deletions} across {changed_files} files"
    )

    data = json.dumps({"content": msg}).encode('utf-8')
    req = urllib.request.Request(
        f"https://discord.com/api/channels/{channel_id}/messages",
        data=data,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        resp.read()


if __name__ == "__main__":
    main()
