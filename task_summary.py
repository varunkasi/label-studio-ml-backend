import os
import requests


def _sort_by_created(items):
    # Assumes ISO 8601 strings, which sort correctly lexicographically
    return sorted(items, key=lambda obj: obj.get("created_at") or "")


def _print_table(items, title):
    print(title)
    print(f"{'ID':<12} {'created_at'}")
    print("-" * 40)
    for obj in items:
        obj_id = obj.get("id", "")
        created = obj.get("created_at", "")
        print(f"{obj_id:<12} {created}")
    print()  # blank line


def print_task_annotations_and_predictions(
    task_id,
    ls_url=None,
    api_key=None,
):
    """
    Fetch a Label Studio task by ID and print sorted annotations and predictions.

    Uses LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY from the environment
    if ls_url or api_key are not explicitly provided.
    """
    if ls_url is None:
        ls_url = (
            os.environ.get("LABEL_STUDIO_URL")
            or os.environ.get("LABEL_STUDIO_HOST")
            or "https://app.heartex.com"
        )

    if api_key is None:
        api_key = os.environ.get("LABEL_STUDIO_API_KEY")

    if not api_key:
        raise RuntimeError(
            "LABEL_STUDIO_API_KEY is not set and no api_key was provided."
        )

    base_url = ls_url.rstrip("/")
    headers = {"Authorization": f"Token {api_key}"}

    resp = requests.get(f"{base_url}/api/tasks/{task_id}", headers=headers, timeout=10)
    resp.raise_for_status()
    task = resp.json()

    annotations = task.get("annotations", [])
    predictions = task.get("predictions", [])

    annotations_sorted = _sort_by_created(annotations)
    predictions_sorted = _sort_by_created(predictions)

    _print_table(annotations_sorted, f"Annotations for task {task_id}")
    _print_table(predictions_sorted, f"Predictions for task {task_id}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <task_id>")
        raise SystemExit(1)

    task_id_arg = int(sys.argv[1])
    print_task_annotations_and_predictions(task_id_arg)
