import os
import json
import urllib.request
from urllib.error import HTTPError, URLError


DEFAULT_WEBHOOK = (
    "https://discord.com/api/webhooks/1027016563854950500/"
    "cAfDngZNXXGn-oPJ3FwivwCFPKefomKdoriD63HHUZ-TZnMagudSX-Zj6aI7cQczai_t"
)


def main():
    webhook_url = os.environ.get("WEBHOOK_URL", DEFAULT_WEBHOOK)


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

    data = json.dumps({"content": msg}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            resp.read()
    except HTTPError as e:
        body = e.read().decode()
        print(f"Discord API responded with {e.code}: {body}")
    except URLError as e:
        print(f"Failed to reach Discord: {e.reason}")


if __name__ == "__main__":
    main()
